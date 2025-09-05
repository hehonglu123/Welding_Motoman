import time, os, copy, sys, yaml, pathlib
import traceback
from copy import deepcopy
import numpy as np
from scipy.interpolate import CubicSpline
import datetime
from motoman_def import *
from lambda_calc import *
import open3d as o3d
import torch
import torch.nn as nn

from RobotRaconteur.Client import *
from weldRRSensor import *
from StreamingSend import *
from robotics_utils import *
sys.path.append('../')
from weld_dh2v import *
sys.path.append('../../scan/scan_process/')
sys.path.append('../../scan/scan_tools/')
from scan_utils import *
from scanProcess import *
from threading import Thread
from controlModelFunction import *

# for plotting
xy_label_size = 18
xy_tick_size = 16
legend_size = 16
title_size = 20
sup_title_size = 20

inch2mm = 25.4
mm2inch = 1/25.4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def welder_handler(exp):
	if (exp is not None):
		# If "err" is not None it means that an exception occurred.
		# "err" contains the exception object
		print ("An error occured! " + str(exp))
		return

def get_target_dh(current_x, last_profile_height, target_layer_height, lookahead_distance, forward):

    if forward:
        lookahead_x = current_x + lookahead_distance
        valid_index = np.where((last_profile_height[:,0]>=current_x) & (last_profile_height[:,0]<=lookahead_x))
    else:
        lookahead_x = current_x - lookahead_distance
        valid_index = np.where((last_profile_height[:,0]<=current_x) & (last_profile_height[:,0]>=lookahead_x))
    if len(valid_index[0]) == 0:
        print('current_x:', current_x, 'lookahead_x:', lookahead_x)
        print("No valid profile height found in the lookahead distance, using the last 20 points")
        next_height = np.mean(last_profile_height[:20,1]) if current_x < 0 else np.mean(last_profile_height[-20:,1])
    else:
        next_height = np.mean(last_profile_height[valid_index,1])
    next_dh = target_layer_height - next_height
    return next_dh

def get_target_dw(current_x, dw_target_profile, lookahead_distance, forward):

    if forward:
        lookahead_x = current_x + lookahead_distance
        valid_index = np.where((dw_target_profile[:,0]>=current_x) & (dw_target_profile[:,0]<=lookahead_x))
    else:
        lookahead_x = current_x - lookahead_distance
        valid_index = np.where((dw_target_profile[:,0]<=current_x) & (dw_target_profile[:,0]>=lookahead_x))
    if len(valid_index[0]) == 0:
        print('current_x:', current_x, 'lookahead_x:', lookahead_x)
        print("No valid dw target found in the lookahead distance, using the last 20 points")
        next_dw = np.mean(dw_target_profile[:20,1]) if current_x < 0 else np.mean(dw_target_profile[-20:,1])
    else:
        next_dw = np.mean(dw_target_profile[valid_index,1])
    return next_dw

def get_weld_shift_x(profile_height):
    profile_x = np.arange(np.min(profile_height[:,0]), np.max(profile_height[:,0])+0.1, 0.1)
    height_approx_func = CubicSpline(profile_height[:,0], profile_height[:,1])
    profile_height_aug = np.column_stack((profile_x, height_approx_func(profile_x)))
    
    # scan_N = 200
    # span_N = 5
    # threshold = 0.1
    # diff_points_1 = []
    # height_diff = np.diff(profile_height_aug[:,1])
    # for point_i, point in enumerate(profile_height_aug[0:scan_N+1]):
    #     diff_right = np.mean(height_diff[point_i:point_i+span_N])

    #     diff_points_1.append(diff_right)
    # # find the first diff points > 0.1
    # left_point = np.argwhere(np.array(diff_points_1) > threshold).flatten()[0]+int(span_N/2)
    # diff_points_2 = []
    # for point_i, point in enumerate(profile_height_aug[::-1][0:scan_N+1]):
    #     diff_right = np.mean(height_diff[::-1][point_i:point_i+span_N])
    #     diff_points_2.append(diff_right)
    # # find the first diff points < -0.1
    # right_point = np.argwhere(np.array(diff_points_2) < -threshold).flatten()[0]+int(span_N/2)

    # left_x = np.mean(profile_height_aug[left_point:left_point+2, 0])
    # right_x = np.mean(profile_height_aug[::-1][right_point:right_point+2, 0])
    # shift_x = -1*(left_x+right_x)/2

    reference_height = 3.5
    profile_height_closed_arg = np.argsort(np.abs(profile_height_aug[:,1]-reference_height))
    left_x = None
    right_x = None
    for profile_idx in profile_height_closed_arg:
        if profile_height_aug[profile_idx,0]<0 and left_x is None:
            left_x = profile_height_aug[profile_idx,0]
        if profile_height_aug[profile_idx,0]>0 and right_x is None:
            right_x = profile_height_aug[profile_idx,0]
        if left_x is not None and right_x is not None:
            break
    shift_x = -1*(left_x+right_x)/2

    # visualize the height
    plt.figure(figsize=(16, 5))
    plt.plot(profile_height_aug[:, 0], profile_height_aug[:, 1], '-o', label='Profile Height')
    # draw a vertical line at left_x and right_x
    plt.axvline(x=left_x, color='r', linestyle='--', label='Left Shift Point')
    plt.axvline(x=right_x, color='g', linestyle='--', label='Right Shift Point')
    plt.title('Profile Height Visualization')
    plt.xlabel('X Position (mm)')
    plt.ylabel('Height (mm)')
    plt.legend()
    plt.grid()
    plt.show()
    
    return shift_x

def main():
    
    weld_arcon = False
    welder_log = False
    fuji_scanon = False
    scan_online_process = False
    thermal_on = False
    input_from_user = False
    SIMULATION = False
    # simulation_speed_sim = False
    simulation_save_control_state_fig = False

    if SIMULATION:
        weld_arcon = False
        welder_log = False
        fuji_scanon = False
        scan_online_process = False
        thermal_on = False

    ############## Robot definition ##############
    config_dir='../../config/'
    robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=10,tool_file_path=config_dir+'torch_robot.csv',\
        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
        base_marker_config_file=config_dir+'MA2010_marker_config/MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config/weldgun_marker_config.yaml')
    robot_scan=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'fujicam.csv',\
        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
    # get fujicam standoff distance
    # Move the fujicam frame along the z-axis with the distance of the standoff distance
    # will locate the frame onto 
    # the plane perpendicular to the weldgun axis and passing through the weldgun TCP
    T_weldgun = robot_weld.fwd(np.zeros(6))
    T_scanner = robot_scan.fwd(np.zeros(6))
    fujicam_standoff_d = np.dot((T_weldgun.p-T_scanner.p),T_weldgun.R[:3,2])/np.dot(T_scanner.R[:3,2],T_weldgun.R[:3,2])
    robot_scan_motion=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=fujicam_standoff_d,tool_file_path=config_dir+'fujicam.csv',\
        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
    robot_thermal=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
	                        pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
    positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
        base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
        base_marker_config_file=config_dir+'D500B_marker_config/D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config/positioner_tcp_marker_config.yaml')
    r_weld_z = robot_weld.fwd(np.zeros(6))
    r_scan_z = robot_scan_motion.fwd(np.zeros(6))
    T_weld_scan = r_weld_z.inv()*r_scan_z
    dist_weld_scan = np.linalg.norm(T_weld_scan.p)
    ########################################################RR STREAMING########################################################
    stream_rate = 125.
    if not SIMULATION:
        # RR_robot_sub = RRN.SubscribeService('rr+tcp://192.168.55.12:59945?service=robot')
        RR_robot_sub = RRN.SubscribeService('rr+tcp://localhost:59945?service=robot')
        point_distance=0.04		###STREAMING POINT INTERPOLATED DISTANCE
        SS=StreamingSend(RR_robot_sub,streaming_rate=stream_rate)
    ########################################################RR FRONIUS########################################################
    if weld_arcon:
        fronius_sub=RRN.SubscribeService('rr+tcp://192.168.55.21:60823?service=welder')
        try:
            fronius_client = fronius_sub.GetDefaultClientWait(1)      #connect, timeout=30s
        except:
            print("Fronius connection failed")

            traceback.print_exc()
            SS.deinitialize_robot()
            exit()
        hflags_const = RRN.GetConstants("experimental.fronius", fronius_client)["WelderStateHighFlags"]
        fronius_client.prepare_welder()
        current_ser=RRN.SubscribeService('rr+tcp://192.168.55.21:12182?service=Current')
    if welder_log and not weld_arcon:
        print("looog")
        fronius_sub=RRN.SubscribeService('rr+tcp://192.168.55.21:60823?service=welder')
        current_ser=RRN.SubscribeService('rr+tcp://192.168.55.21:12182?service=Current')
    
    ######################################### RR Fujicam ########################################################
    if fuji_scanon:
        fujicam_url = 'rr+tcp://localhost:12181/?service=fujicam'
        def connect_failed(s, client_id, url, err):
            print ("Client connect failed: " + str(client_id.NodeID) + " url: " + str(url) + " error: " + str(err))
        sub=RRN.SubscribeService(fujicam_url)
        obj = sub.GetDefaultClientWait(2)		#connect, timeout=2s
        fuji_scan_wire=sub.SubscribeWire("lineProfile")
        sub.ClientConnectFailed += connect_failed

        scan_process = ScanProcess(robot_scan,positioner) # initialize scan process

    ########################################## RR Thermal ########################################################
    if thermal_on:
        flir_url = 'rr+tcp://192.168.55.10:60827/?service=camera'
        cam_ser=RRN.ConnectService(flir_url)
        if weld_arcon or welder_log:
            rr_sensors = WeldRRSensor(weld_service=fronius_sub,cam_service=cam_ser,current_service=current_ser)
        else:
            rr_sensors = WeldRRSensor(cam_service=cam_ser)
        # print("Test 3 Sec.")
        # rr_sensors.test_all_sensors()
        # print(len(rr_sensors.ir_recording))
        # rr_sensors.save_all_sensors('')
        # if weld_arcon:
        #     fronius_client.release_welder()
        # exit()

    ################## Read geometry data ##################
    data_dir = '../../data/wall_weld_test/'
    with open(data_dir+'sliced_meta.yml', 'r') as f:
        meta_data = yaml.safe_load(f)
    
    base_layer_num= meta_data['baselayer_num']
    baselayer_resolution= meta_data['baselayer_resolution']
    layer_num = meta_data['layer_num']
    layer_resolution = meta_data['layer_resolution']
    path_dl = meta_data['path_dl']
    dist_weld_scan_index = np.round(dist_weld_scan/path_dl).astype(int)

    # material_name = 'ER4043'
    material_name = 'ER316L'

    if material_name == 'ER4043':
        job_offset=200
        # feedrate min max (based on material ER4043)
        feedrate_min = 100 # inch/min
        feedrate_max = 200 # inch/min
        # baselayer welding parameters
        base_feedrate = 250 
        base_nom_incre = 1
        base_nom_vel = 5
        # layer welding parameters
        layer_feedrate = 100 # inch/min 
        layer_nom_height = 3 # mm
        layer_nom_vel = 5*np.sqrt(2) # mm/s => 1, 1/np.sqrt(2), 1/2, np.sqrt(2), 2, affecting VPD
        layer_nom_incre = int(layer_nom_height/layer_resolution)
        # wire cross section
        cross_section = 1.2 # mm^2
    elif material_name == 'ER316L':
        job_offset=450
        # feedrate min max (based on material ER316L)
        feedrate_min = 50 # inch/min
        feedrate_max = 250 # inch/min
        # baselayer welding parameters
        base_feedrate = 300 
        base_nom_incre = 1
        base_nom_vel = 5
        # layer welding parameters
        tune_ratio = 1.5
        layer_feedrate = tune_ratio*100
        layer_nom_vel = tune_ratio*10*1/2 # mm/s => 1, 1/np.sqrt(2), 1/2, 1/(2*np.sqrt(2)), 1/4, affecting VPD
        layer_nom_height = 3 # mm
        layer_nom_incre = int(layer_nom_height/layer_resolution)
        # wire cross section
        cross_section = 1.14 # mm^2
    if SIMULATION:
        base_nom_vel = 50 # for speed up
    
    ##### motion parameters #####
    # streaming rate
    feedrate_update_rate=10	#Hz
    # weld starting point sleep
    weld_start_sleep = 0.2
    # scanning parameters
    scan_nom_vel = 8
    # collision avoidance z offset
    safety_z_offset = 50
    # direction 
    torch_ori_fix = True # torch orientation fixed
    # lookahead distance
    lookahead_distance = 1 # mm
    # which layer to start correction
    correction_layer_start = 2 # start correction from layer 2, set to a large number if no correction layer
    # correction_layer_start = 99999999999999 # no correction layer, set to a large number

    ##### controller parameters and model #####
    ### Learning model
    control_model_dir = 'model_20250715_151650'
    ctrlModel = controlModel(control_model_dir,device=device)
    alpha_control = 0.25
    lambda_smooth = 1e-2  # regularization parameter for smoothness
    lambda_disc = 1e-1*5  # regularization parameter for discrete input
    ### log-log model
    loglog_model_dir = 'loglog_models'
    loglogModel = controlLogLogModel(loglog_model_dir)
    # max min torch velocity
    v_Maximum = 20
    v_minimum = 0.75
    # choose between log-log control or learning model one step Jacobian
    control_method = 'loglog-rls' # 'loglog-static', 'loglog-rls' or 'learning-Jacobian'
    assert control_method in ['loglog-static', 'loglog-rls', 'learning-Jacobian'], "Invalid control method"
    #######################################

    ##### welding target parameters #####
    curve = np.loadtxt(data_dir+f'curve_sliced_relative/slice0_0.csv',delimiter=',')
    curve_x_start = np.min(curve[:,0])
    curve_x_end = np.max(curve[:,0])
    curve_x_sample = np.arange(curve_x_start, curve_x_end+0.1, 0.1) # sample points for the curve

    weld_type = 'static' # 'static' or 'axe'
    ## static dw
    if weld_type == 'static':
        dh_target, dw_target_singlePoint = loglogModel.get_pred_loglog(layer_nom_vel, layer_feedrate) # get the target dh and dw from the control loglog
        dw_target = np.vstack((curve_x_sample, np.ones_like(curve_x_sample)*dw_target_singlePoint)).T # create a constant dw target for the whole layer
    elif weld_type == 'axe':
        ## axe like dw (thick to thin)
        dh_target, _ = loglogModel.get_pred_loglog(layer_nom_vel, layer_feedrate)
        dw_target_large = 6.25
        dw_target_small = 3.25
        dw_target = np.vstack((curve_x_sample, np.linspace(dw_target_large, dw_target_small, len(curve_x_sample)))).T # create a dw target that decreases from large to small
    else:
        assert False, "Invalid weld type, must be 'static' or 'axe'"

    print(f'Target dh: {dh_target:.2f} mm, dw: {np.mean(dw_target[:,1]):.2f} mm')
    ########################################

    ##### visualization parameters #####
    viz_interval = 1 # visualize every N sec
    ####################################
    
    ##### Log data dir #####
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-7]
    logdata_dir='../../data/wall_weld_test/weld_fujicontrol_'+formatted_time+'/'
    # logdata_dir='../../data/wall_weld_test/weld_fujicontrol_2025_08_13_14_17_58/'

    ##### Parameters to chose where to start welding #####
    # start-end layers
    baselayer_start = 0
    baselayer_end = base_layer_num
    layer_start = 0
    layer_end = layer_num
    # read from file or not
    read_from_file_layer = False
    Transz0_H=None
    last_profile_height = None
    if read_from_file_layer:
        logdata_dir = '../../data/wall_weld_test/weld_fujicontrol_2025_08_13_14_17_58/'
        Transz0_H = np.array([[ 9.99850748e-01 , 1.22093145e-04 , 1.72761855e-02 ,-1.41469281e-01],
                    [ 1.22093145e-04 , 9.99900124e-01 ,-1.41325105e-02,  1.15726710e-01],
                    [-1.72761855e-02,  1.41325105e-02,  9.99750872e-01, -8.18664727e+00],
                    [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])
        last_profile_height = None
    # logdata_dir='../../data/wall_weld_test/weld_fujicontrol_2025_08_13_14_17_58/'
    # Transz0_H = np.array([[ 9.99850748e-01 , 1.22093145e-04 , 1.72761855e-02 ,-1.41469281e-01],
    #                 [ 1.22093145e-04 , 9.99900124e-01 ,-1.41325105e-02,  1.15726710e-01],
    #                 [-1.72761855e-02,  1.41325105e-02,  9.99750872e-01, -8.18664727e+00],
    #                 [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])
    # last_profile_height = np.loadtxt(logdata_dir+'baselayer1/profile_height.csv', delimiter=',') # load the last profile height
    #################################################3

    ##### weld meta data #####
    weld_meta_data = {'well_arcon':weld_arcon, 'fuji_scanon':fuji_scanon, 'data_dir':data_dir, 'logdata_dir':logdata_dir\
                      ,'material_name':material_name\
                    ,'base_layer_num':base_layer_num, 'baselayer_resolution':baselayer_resolution, 'layer_num':layer_num, 'layer_resolution':layer_resolution\
                    ,'base_feedrate':base_feedrate, 'base_nom_incre':base_nom_incre, 'base_nom_vel':base_nom_vel\
                    ,'layer_feedrate':layer_feedrate, 'layer_nom_incre':layer_nom_incre, 'layer_nom_vel':float(round(layer_nom_vel,3))\
                    ,'cross_section':cross_section, 'dh_target':float(dh_target), 'dw_target':float(np.mean(dw_target[:,1])), 'lookahead_distance':lookahead_distance,\
                    'alpha_control':alpha_control, 'lambda_smooth':lambda_smooth, 'lambda_disc':lambda_disc,\
                    'model_dir':control_model_dir, 'v_Maximum':v_Maximum, 'v_minimum':v_minimum, 'weld_type':weld_type,\
                    'loglog_model_dir':loglog_model_dir, 'control_method':control_method}
    ##############################

    ####### simulation setup #####
    if SIMULATION or not weld_arcon:
        sim_folder = '../../data/wall_weld_test/weld_fujicontrol_2025_08_14_11_19_59/'
        total_layers_name = glob.glob(sim_folder+'layer*')
        # get printed layer number
        layer_nums = []
        for layer_name in total_layers_name:
            this_layer = layer_name.split('\\')[-1]
            this_layer = this_layer.split('r')[-1]
            layer_nums.append(int(this_layer))
        layer_nums = np.sort(layer_nums)
    ####################################################3

    ##### Welding ready to start #####
    print("Logged Data Dir:",logdata_dir)
    print("Material Name:",material_name)
    print("Weld Type:",weld_type)
    input("Ready to start? Press Enter to continue...")
    ################## print layers ##################
    arc_off=True
    forward = True

    mean_layer_height = 0
    for weld_parts in ['base','layer']:
    # for weld_parts in ['layer']:
        if weld_parts == 'base':
            weld_start = baselayer_start
            # weld_start = 1
            weld_end = baselayer_end
            nom_incre = base_nom_incre
        else:
            weld_start = layer_start
            weld_end = layer_end
            # weld_end = 36
            nom_incre = layer_nom_incre
            if weld_start == 0:
                shift_weld_profile_x = get_weld_shift_x(last_profile_height)
                # shift_weld_profile_x = 5.75
                print("Shift Weld Profile X:", shift_weld_profile_x)

        layer_count = 0
        i=weld_start
        print("Welding parts:",weld_parts)
        print("Start layer:",weld_start,"End layer:",weld_end,"Nominal Increment:",nom_incre)
        input("Press Enter to continue...")
        while i < weld_end:
            if weld_parts == 'layer':
                # compensate the shift of the weld profile
                last_profile_height[:,0] = last_profile_height[:,0] + shift_weld_profile_x
                target_layer_height = dh_target + mean_layer_height
            print("=====================================")
            print(f'Welding {weld_parts} layer {i} counting {layer_count} direction {forward}')
            try:
                if torch_ori_fix:
                    print("Torch Orientation Fixed")
                    curve_direction = 'backward'
                elif forward:
                    curve_direction = 'forward'
                else:
                    curve_direction = 'backward'
                # read curve joint space data
                if weld_parts == 'base':
                    curve = np.loadtxt(data_dir+f'curve_sliced_relative/baselayer{i}_0.csv',delimiter=',')
                    curve_scan = np.loadtxt(data_dir+f'curve_sliced_relative/baselayer{i}_0_scan_{curve_direction}.csv',delimiter=',')
                    curve_js = np.loadtxt(data_dir+f'curve_sliced_js/MA2010_base_js{i}_0_{curve_direction}.csv', delimiter=',')
                    curve_js_cam = np.loadtxt(data_dir+f'curve_sliced_js/MA1440_base_js{i}_0_{curve_direction}.csv', delimiter=',')
                    curve_js_positioner = np.loadtxt(data_dir+f'curve_sliced_js/D500B_base_js{i}_0_{curve_direction}.csv', delimiter=',')
                    curve_js_scan = np.loadtxt(data_dir+f'curve_sliced_js/MA2010_base_js{i}_0_scan_{curve_direction}.csv', delimiter=',')
                    curve_js_pos_scan = np.loadtxt(data_dir+f'curve_sliced_js/D500B_base_js{i}_0_scan_{curve_direction}.csv', delimiter=',')
                else:
                    curve = np.loadtxt(data_dir+f'curve_sliced_relative/slice{i}_0.csv',delimiter=',')
                    curve_scan = np.loadtxt(data_dir+f'curve_sliced_relative/slice{i}_0_scan_{curve_direction}.csv',delimiter=',')
                    curve_js = np.loadtxt(data_dir+f'curve_sliced_js/MA2010_js{i}_0_{curve_direction}.csv', delimiter=',')
                    curve_js_cam = np.loadtxt(data_dir+f'curve_sliced_js/MA1440_js{i}_0_{curve_direction}.csv', delimiter=',')
                    curve_js_positioner = np.loadtxt(data_dir+f'curve_sliced_js/D500B_js{i}_0_{curve_direction}.csv', delimiter=',')
                    curve_js_scan = np.loadtxt(data_dir+f'curve_sliced_js/MA2010_js{i}_0_scan_{curve_direction}.csv', delimiter=',')
                    curve_js_pos_scan = np.loadtxt(data_dir+f'curve_sliced_js/D500B_js{i}_0_scan_{curve_direction}.csv', delimiter=',')
                
                if not forward:
                    curve = curve[::-1]
                lam_relative = calc_lam_cs(curve[:,:3])
                lam_scan_relative = calc_lam_cs(curve_scan[:,:3])

                if forward and curve_direction == 'backward':
                    curve_js = curve_js[::-1]
                    curve_js_cam = curve_js_cam[::-1]
                    curve_js_positioner = curve_js_positioner[::-1]
                
                if not read_from_file_layer: # actually weld a layer

                    if input_from_user:
                        input("Press Enter to continue...")
                    
                    ### turn on sensors
                    if thermal_on:
                        rr_sensors.start_all_sensors()

                    # move to start point with safety_z_offset
                    if not SIMULATION:
                        for z in np.arange(safety_z_offset,0,-5): # a linear movement
                            T_start = robot_weld.fwd(curve_js[0])
                            T_start.p[2] += z
                            curve_js_start_offset = robot_weld.inv(T_start.p, T_start.R, last_joints=curve_js[0])[0]
                            q_start_offset = np.hstack((curve_js_start_offset, curve_js_cam[0], curve_js_positioner[0]))
                            SS.jog2q(q_start_offset)
                            if z==safety_z_offset:
                                time.sleep(0.1)
                        # move to start point
                        q_start = np.hstack((curve_js[0], curve_js_cam[0], curve_js_positioner[0]))
                        SS.jog2q(q_start) 
                        time.sleep(0.1) # wait for the robot to reach the start point, clean the buffer

                    ##### welding motion #####
                    lam_cur=0
                    cmd_update_cnt = 0 # command update count
                    q_cmd_all = [] # log joint space command
                    welding_cmd_all = [] # log welding command
                    weld_js_exe = [] # log executed joint space
                    scan_exe = [] # log scan data
                    if scan_online_process: # if online scan processing is enabled
                        scan_exe_noise_remove = []
                        scan_denoise_thread = Thread(target=scan_process.scan_denoise_thread, args=([-40, 30],[40, 200])) # arges: (crop_min, crop_max)
                        scan_denoise_thread.start()
                    
                    ### initial velocity
                    if weld_parts == 'base':
                        v_cmd = base_nom_vel
                        feedrate_cmd = base_feedrate
                    else:
                        current_x = curve[0][0]
                        next_dw = get_target_dw(current_x, dw_target, lookahead_distance, forward)
                        if layer_count < correction_layer_start: #
                            next_dh = dh_target
                        else:
                            next_dh = get_target_dh(current_x, last_profile_height, target_layer_height, lookahead_distance, forward)
                        v_cmd ,feedrate_cmd = loglogModel.get_control_loglog(next_dh,next_dw) # get the velocity and feedrate from the control loglog as the initial
                        dh_pred, dw_pred = loglogModel.get_pred_loglog(v_cmd, feedrate_cmd) # get the predicted dh and dw from the control loglog
                        print(f'Initial Torch V: {v_cmd:.2f} mm/s, Feedrate: {feedrate_cmd:.2f} inch/min, dh_pred: {dh_pred:.2f} mm, dw_pred: {dw_pred:.2f} mm')
                    
                    ### start welding and data logging
                    # if SIMULATION:
                    control_status_log = []
                    time_count = []
                    model_inference_time_count = [0,0]
                    while lam_cur < (lam_relative[-1] - v_cmd/stream_rate):
                        loop_start=time.perf_counter()

                        ### get the next lambda idx
                        lam_cur+=v_cmd/stream_rate # get the current lambda (path location)
                        lam_idx=np.where(lam_relative>=lam_cur)[0][0] #get closest two indices and interpolate the joint angle
                        ### get the next q commands
                        ratio=(lam_cur-lam_relative[lam_idx-1])/(lam_relative[lam_idx]-lam_relative[lam_idx-1]) # find the ratio for interpolation
                        this_curve_p = curve[lam_idx-1]*(1-ratio)+curve[lam_idx]*ratio
                        q1=curve_js[lam_idx-1]*(1-ratio)+curve_js[lam_idx]*ratio # robot 1 joint angles
                        q2=curve_js_cam[lam_idx-1]*(1-ratio)+curve_js_cam[lam_idx]*ratio # robot 2 joint angles
                        q_pos=curve_js_positioner[lam_idx-1]*(1-ratio)+curve_js_positioner[lam_idx]*ratio # positioner joint angles
                        q_cmd=np.hstack((q1,q2,q_pos)) # command joint angles (combined robot 1, robot 2 and positioner)

                        ### if welding start
                        if arc_off:
                            if weld_arcon:
                                print("Welding Start")
                                fronius_client.job_number = int(round(feedrate_cmd/10)+job_offset) # get fronius job number
                                fronius_client.start_weld() # command to start welding
                                time.sleep(weld_start_sleep) # welder needs about 0.2s to start welding
                            welding_cmd_all.append(np.hstack((time.perf_counter(),i,this_curve_p[0],v_cmd,int(round(feedrate_cmd/10)*10))))
                            last_update_time=time.perf_counter()
                            last_viz_time=time.perf_counter()
                            cmd_update_cnt += 1
                            arc_off=False

                        ### update welding param
                        weld_cmd_updated = False
                        if time.perf_counter()-last_update_time>1./feedrate_update_rate:
                            model_inference_start_time = time.perf_counter()
                            if weld_parts == 'layer':
                                # update feedrate to welder
                                current_x = this_curve_p[0]
                                next_dw = get_target_dw(current_x, dw_target, lookahead_distance, forward)
                                if layer_count < correction_layer_start:
                                    next_dh = dh_target
                                else:
                                    next_dh = get_target_dh(current_x, last_profile_height, target_layer_height, lookahead_distance, forward)
                                if control_method == 'learning-Jacobian':
                                    v_cmd, feedrate_cmd, dh_pred, dw_pred = ctrlModel.forward_one_step_get_opt_u(v_cmd, feedrate_cmd, next_dh, next_dw, alpha_control,  lambda_smooth=lambda_smooth, lambda_disc=lambda_disc)
                                elif 'loglog' in control_method:
                                    v_cmd ,feedrate_cmd = loglogModel.get_control_loglog(next_dh,next_dw)
                                    dh_pred, dw_pred = loglogModel.get_pred_loglog(v_cmd, feedrate_cmd)
                                v_cmd = np.clip(v_cmd, v_minimum, v_Maximum) # clip the velocity
                                weld_cmd_updated = True
                            if weld_arcon:
                                fronius_client.async_set_job_number(int(round(feedrate_cmd/10)+job_offset), welder_handler)
                            model_inference_time_count.append(time.perf_counter()-model_inference_start_time)
                            # log command data
                            welding_cmd_all.append(np.hstack((time.perf_counter(),i,this_curve_p[0],v_cmd,int(round(feedrate_cmd/10)*10))))
                            last_update_time=time.perf_counter()
                            if (time.perf_counter()-last_viz_time)>viz_interval:
                                print("Current X:", this_curve_p[0], "Update Feedrate, Velocity:",int(round(feedrate_cmd/10)*10),round(v_cmd,1))
                                last_viz_time=time.perf_counter()
                        if weld_parts == 'layer':
                            control_status_log.append(np.hstack((time.perf_counter(),this_curve_p[0],v_cmd,int(round(feedrate_cmd/10)*10),next_dh, next_dw, dh_pred, dw_pred, weld_cmd_updated)))
                        
                        ### log data, line scanner (fujicam), robot welding joints
                        if fuji_scanon:
                            wire_packet=fuji_scan_wire.TryGetInValue() # log fuji cam scanner data
                            valid_indices=np.where(wire_packet[1].I_data>1)[0]
                            valid_indices=np.intersect1d(valid_indices,np.where(np.abs(wire_packet[1].Z_data)>50)[0])
                            line_profile=np.hstack((wire_packet[1].Y_data[valid_indices].reshape(-1,1),wire_packet[1].Z_data[valid_indices].reshape(-1,1)))
                            scan_exe.append(line_profile)
                        weld_js_exe.append(np.append(time.perf_counter(),deepcopy(SS.q_cur))) if not SIMULATION else None # log timestamp and robot joints

                        ### scan online denoising
                        if fuji_scanon and scan_online_process:
                            scan_process.raw_scan_pipe.append(deepcopy(line_profile))
                            while len(scan_process.denoise_pipe)!=0:
                                scan_denoise = scan_process.denoise_pipe.pop(0)
                                scan_exe_noise_remove.append(scan_denoise)

                        ### sent position Command to the robot
                        q_cmd_all.append(np.hstack((time.perf_counter(),i,q_cmd)))
                        if not SIMULATION:
                            SS.position_cmd(q_cmd,loop_start)
                        else:
                            while time.perf_counter()-loop_start < 1/stream_rate*0.975:
                                time.sleep(0)	#sleep 0 for bg thread to run
                                continue 
                        
                        time_count.append(time.perf_counter()-loop_start) # log time count

                    ##################################################
                    print(f'Mean time per command: {np.mean(time_count):.4f} s, Max time per command: {np.max(time_count):.4f} s')
                    print(f'Mean model inference time: {np.mean(model_inference_time_count):.4f} s, Max model inference time: {np.max(model_inference_time_count):.4f} s')

                    ### welding end
                    if weld_arcon:
                        fronius_client.stop_weld()
                    arc_off=True # turn the arc off
                    ### log the remaining fujicam data
                    if not SIMULATION:
                        fuji_scan_time = 0.5 # stay for a while for scanning, and robot to move to the final position
                        fuji_scan_start = time.perf_counter()
                        while time.perf_counter()-fuji_scan_start<fuji_scan_time:
                            ### log data
                            if fuji_scanon:
                                wire_packet=fuji_scan_wire.TryGetInValue() # log fuji cam scanner data
                                valid_indices=np.where(wire_packet[1].I_data>1)[0]
                                valid_indices=np.intersect1d(valid_indices,np.where(np.abs(wire_packet[1].Z_data)>30)[0])
                                line_profile=np.hstack((wire_packet[1].Y_data[valid_indices].reshape(-1,1),wire_packet[1].Z_data[valid_indices].reshape(-1,1)))
                                scan_exe.append(line_profile)
                            weld_js_exe.append(np.append(time.perf_counter(),deepcopy(SS.q_cur))) if not SIMULATION else None # log robot joints

                            ### scan online processing
                            if fuji_scanon and scan_online_process:
                                scan_process.raw_scan_pipe.append(deepcopy(line_profile))
                                while len(scan_process.denoise_pipe)!=0:
                                    scan_denoise = scan_process.denoise_pipe.pop(0)
                                    # get denoise scan
                                    scan_exe_noise_remove.append(scan_denoise)
                            time.sleep(1/stream_rate)
                    ########################################

                    ### simulation visualization
                    if weld_parts == 'layer' and SIMULATION:
                        profile_height = np.loadtxt(sim_folder+'layer'+str(i)+f'/profile_height.csv',delimiter=',')
                        profile_height_shift = deepcopy(profile_height)
                        profile_height_shift[:,0] += shift_weld_profile_x
                        profile_dh = []
                        for x_id, x_pos in enumerate(profile_height_shift[:,0]):
                            if x_pos>=np.min(curve[:,0]) and x_pos<=np.max(curve[:,0]):
                                last_x_index = np.argmin(np.abs(last_profile_height[:, 0] - x_pos))
                                profile_dh.append([x_pos, profile_height_shift[x_id, 1] - last_profile_height[last_x_index, 1]])
                        profile_dh = np.array(profile_dh)
                        try:
                            profile_width = np.loadtxt(sim_folder+'layer'+str(i)+f'/profile_width.csv',delimiter=',')
                            profile_width_shift = deepcopy(profile_width)
                            profile_width_shift[:,0] += shift_weld_profile_x
                            valid_index = np.where((profile_width_shift[:, 0] >= np.min(curve[:, 0])) & (profile_width_shift[:, 0] <= np.max(curve[:, 0])))
                            profile_width_shift = profile_width_shift[valid_index]
                        except FileNotFoundError:
                            profile_width = None
                        control_status_log = np.array(control_status_log)
                        fig, axs = plt.subplots(3, 1, figsize=(16, 10))
                        mng = plt.get_current_fig_manager()
                        mng.window.state('zoomed')
                        # Adjust subplot spacing/margins
                        fig.subplots_adjust(left=0.07, right=0.95, top=0.912, bottom=0.064, wspace=0.35, hspace=0.545)
                        for ax_idx in range(3):
                            ax = axs[ax_idx]
                            if ax_idx == 2:
                                ax_right = ax.twinx()
                                line1 = ax.plot(control_status_log[:, 1], control_status_log[:, 2], 'g-', label='Torch v (mm)')
                                line2 = ax_right.plot(control_status_log[:, 1], control_status_log[:, 3], 'r-', label='Wire Feed Rate (ipm)')
                                lines = line1 + line2
                                labels = [l.get_label() for l in lines]
                                ax.set_ylabel("mm", fontsize=xy_label_size)
                                ax_right.set_ylabel("ipm", fontsize=xy_label_size)
                                ax_right.tick_params(axis='y', labelsize=xy_tick_size)
                                ax.legend(lines, labels, loc='upper right', fontsize=legend_size)
                                ax.set_title(f"Commanded Torch Velocity & Wire Feed Rate (mm & ipm)", fontsize=title_size)
                            elif ax_idx == 0:
                                ax.plot(control_status_log[:, 1], control_status_log[:, 4], label=f'Target $\Delta h$')                            
                                ax.plot(control_status_log[:, 1], control_status_log[:, 6], label=f'Predicted $\Delta h$')
                                ax.plot(profile_dh[:, 0], profile_dh[:, 1], label=f'Actual $\Delta h$')
                                ax.legend(fontsize=legend_size)
                                ax.set_ylabel("mm", fontsize=xy_label_size)
                                ax.set_title(f"Target vs Predicted vs Actual $\Delta h$ (mm)", fontsize=title_size)
                            elif ax_idx == 1:
                                ax.plot(control_status_log[:, 1], control_status_log[:, 5], label=f'Target Width')                            
                                ax.plot(control_status_log[:, 1], control_status_log[:, 7], label=f'Predicted Width')
                                ax.plot(profile_width_shift[:, 0], profile_width_shift[:, 1], label=f'Actual Width')
                                ax.legend(fontsize=legend_size)
                                ax.set_ylabel("mm", fontsize=xy_label_size)
                                ax.set_title(f"Target vs Predicted vs Actual Width (mm)", fontsize=title_size)
                            # ax.set_xlabel("Time (s)")
                            ax.set_xlabel("X (mm)", fontsize=xy_label_size)
                            ax.tick_params(axis='both', which='major', labelsize=xy_tick_size)
                            ax.grid(True)
                        plt.suptitle(f"Layer {layer_count} Control Status", fontsize=title_size)
                        # plt.show()
                        if simulation_save_control_state_fig:
                            fig.savefig(sim_folder+'layer'+str(i)+'/control_status.png', dpi=300, bbox_inches='tight')
                            plt.close(fig)
                        else:
                            plt.show()

                    # if torch orientation is fixed, and the traveling/curve direction is opposite
                    if forward and curve_direction == 'backward':
                        # deal with special case, forward but curve direction is backward
                        # happens if fixed torch orientation
                        scan_layer = int(np.min([i+6/layer_resolution,weld_end-1]))
                        # read curve joint space data
                        if weld_parts == 'base':
                            curve_dummy = np.loadtxt(data_dir+f'curve_sliced_relative/baselayer{scan_layer}_0.csv',delimiter=',')
                            curve_scan = np.loadtxt(data_dir+f'curve_sliced_relative/baselayer{scan_layer}_0_scan_{curve_direction}.csv',delimiter=',')
                            curve_js = np.loadtxt(data_dir+f'curve_sliced_js/MA2010_base_js{scan_layer}_0_{curve_direction}.csv', delimiter=',')
                            curve_js_positioner = np.loadtxt(data_dir+f'curve_sliced_js/D500B_base_js{scan_layer}_0_{curve_direction}.csv', delimiter=',')
                            curve_js_scan = np.loadtxt(data_dir+f'curve_sliced_js/MA2010_base_js{scan_layer}_0_scan_{curve_direction}.csv', delimiter=',')
                            curve_js_pos_scan = np.loadtxt(data_dir+f'curve_sliced_js/D500B_base_js{scan_layer}_0_scan_{curve_direction}.csv', delimiter=',')
                        else:
                            curve_dummy = np.loadtxt(data_dir+f'curve_sliced_relative/slice{scan_layer}_0.csv',delimiter=',')
                            curve_scan = np.loadtxt(data_dir+f'curve_sliced_relative/slice{scan_layer}_0_scan_{curve_direction}.csv',delimiter=',')
                            curve_js = np.loadtxt(data_dir+f'curve_sliced_js/MA2010_js{scan_layer}_0_{curve_direction}.csv', delimiter=',')
                            curve_js_positioner = np.loadtxt(data_dir+f'curve_sliced_js/D500B_js{scan_layer}_0_{curve_direction}.csv', delimiter=',')
                            curve_js_scan = np.loadtxt(data_dir+f'curve_sliced_js/MA2010_js{scan_layer}_0_scan_{curve_direction}.csv', delimiter=',')
                            curve_js_pos_scan = np.loadtxt(data_dir+f'curve_sliced_js/D500B_js{scan_layer}_0_scan_{curve_direction}.csv', delimiter=',')
                        curve_dummy = curve_dummy[::-1]
                        curve_scan = curve_scan[::-1]
                        curve_dummy = curve_dummy[int(dist_weld_scan_index-cross_section/path_dl):]
                        curve_js = curve_js[int(dist_weld_scan_index-cross_section/path_dl):]
                        curve_js_positioner = curve_js_positioner[int(dist_weld_scan_index-cross_section/path_dl):]
                        curve_js_scan = np.vstack((curve_js,curve_js_scan))
                        curve_js_pos_scan = np.vstack((curve_js_positioner,curve_js_pos_scan))
                        lam_scan_relative = calc_lam_cs(curve_dummy[:,:3])
                        lam_scan_relative = np.append(lam_scan_relative,calc_lam_cs(curve_scan[:,:3])+lam_scan_relative[-1])
                        # move to start point with safety_z_offset
                        if not SIMULATION:
                            q_cur = deepcopy(SS.q_cur)
                            for z in np.arange(0,safety_z_offset+1,5): # a linear movement
                                T_end = robot_weld.fwd(q_cur[:6])
                                T_end.p[2] += z
                                curve_js_end_offset = robot_weld.inv(T_end.p, T_end.R, last_joints=q_cur[:6])[0]
                                q_end_offset = np.hstack((curve_js_end_offset, q_cur[6:]))
                                SS.jog2q(q_end_offset)
                            # move to start point
                            q_start = np.hstack((curve_js_scan[0], q2, curve_js_pos_scan[0]))
                            SS.jog2q(q_start)
                        
                    ####### remain scanning motion ##########################
                    r2_rest_q = q2
                    v_cmd = scan_nom_vel
                    lam_cur=0
                    if not SIMULATION:
                        while lam_cur<lam_scan_relative[-1] - v_cmd/stream_rate:
                            loop_start=time.perf_counter()

                            ### get the next q commands
                            lam_cur+=v_cmd/stream_rate # get the current lambda (path location)
                            lam_idx=np.where(lam_scan_relative>=lam_cur)[0][0] # get closest two indices and interpolate the joint angle
                            ratio=(lam_cur-lam_scan_relative[lam_idx-1])/(lam_scan_relative[lam_idx]-lam_scan_relative[lam_idx-1]) # find the ratio for interpolation
                            q1=curve_js_scan[lam_idx-1]*(1-ratio)+curve_js_scan[lam_idx]*ratio # robot 1 joint angles
                            q_pos=curve_js_pos_scan[lam_idx-1]*(1-ratio)+curve_js_pos_scan[lam_idx]*ratio # positioner joint angles
                            q_cmd=np.hstack((q1,r2_rest_q,q_pos)) # command joint angles (combined robot 1, robot 2 and positioner)

                            ### log data, line scanner (fujicam), robot welding joints
                            if fuji_scanon:
                                wire_packet=fuji_scan_wire.TryGetInValue() # log fuji cam scanner data
                                valid_indices=np.where(wire_packet[1].I_data>1)[0]
                                valid_indices=np.intersect1d(valid_indices,np.where(np.abs(wire_packet[1].Z_data)>50)[0])
                                line_profile=np.hstack((wire_packet[1].Y_data[valid_indices].reshape(-1,1),wire_packet[1].Z_data[valid_indices].reshape(-1,1)))
                                scan_exe.append(line_profile)
                            weld_js_exe.append(np.append(time.perf_counter(),deepcopy(SS.q_cur))) if not SIMULATION else None # log robot joints

                            ### scan online denoising
                            if fuji_scanon and scan_online_process:
                                scan_process.raw_scan_pipe.append(deepcopy(line_profile))
                                while len(scan_process.denoise_pipe)!=0:
                                    scan_denoise = scan_process.denoise_pipe.pop(0)
                                    scan_exe_noise_remove.append(scan_denoise)

                            ### sent position Command to the robot
                            q_cmd_all.append(np.hstack((time.perf_counter(),i,q_cmd)))
                            if not SIMULATION:
                                SS.position_cmd(q_cmd,loop_start)
                            else:
                                time.sleep(1/stream_rate) # wait for the robot to reach the start point, clean the buffer
                                
                        ########################################
                        fuji_scan_time = 0.5 # stay for a while for scanning, and robot to move to the final position
                        fuji_scan_start = time.perf_counter()
                        while time.perf_counter()-fuji_scan_start<fuji_scan_time:
                            ### log data
                            if fuji_scanon:
                                wire_packet=fuji_scan_wire.TryGetInValue() # log fuji cam scanner data
                                valid_indices=np.where(wire_packet[1].I_data>1)[0]
                                valid_indices=np.intersect1d(valid_indices,np.where(np.abs(wire_packet[1].Z_data)>30)[0])
                                line_profile=np.hstack((wire_packet[1].Y_data[valid_indices].reshape(-1,1),wire_packet[1].Z_data[valid_indices].reshape(-1,1)))
                                scan_exe.append(line_profile)
                            weld_js_exe.append(np.append(time.perf_counter(),deepcopy(SS.q_cur))) if not SIMULATION else None # log robot joints

                            ### scan online processing
                            if fuji_scanon and scan_online_process:
                                scan_process.raw_scan_pipe.append(deepcopy(line_profile))
                                while len(scan_process.denoise_pipe)!=0:
                                    scan_denoise = scan_process.denoise_pipe.pop(0)
                                    # get denoise scan
                                    scan_exe_noise_remove.append(scan_denoise)
                            time.sleep(1/stream_rate)

                        # final scan processing
                        if fuji_scanon and scan_online_process:
                            while len(scan_process.raw_scan_pipe)!=0:
                                print("Final scan processing...",len(scan_process.raw_scan_pipe))
                                time.sleep(0.01)
                            while len(scan_process.denoise_pipe)!=0:
                                scan_denoise = scan_process.denoise_pipe.pop(0)
                                scan_exe_noise_remove.append(scan_denoise)
                            # stop scan process
                            scan_process.end_denoise_thread_flag = True
                            scan_denoise_thread.join()

                    # move to end point with safety_z_offset
                    if not SIMULATION:
                        for z in np.arange(0,safety_z_offset+1,5): # a linear movement
                            T_end = robot_weld.fwd(curve_js_scan[-1])
                            T_end.p[2] += z
                            curve_js_end_offset = robot_weld.inv(T_end.p, T_end.R, last_joints=curve_js_scan[-1])[0]
                            q_end_offset = np.hstack((curve_js_end_offset, r2_rest_q, q_pos))
                            SS.jog2q(q_end_offset)

                    ############## save data ######################
                    if weld_parts == 'base':
                        layer_name = 'baselayer'+str(i)
                    else:
                        layer_name = 'layer'+str(i)
                    if not SIMULATION:
                        if not os.path.exists(logdata_dir):
                            os.makedirs(logdata_dir)
                        # save meta data
                        with open(logdata_dir+'weld_meta_data.yml', 'w') as f:
                            yaml.dump(weld_meta_data, f)
                        pathlib.Path(logdata_dir+layer_name).mkdir(parents=True, exist_ok=True)
                        np.savetxt(logdata_dir+layer_name+f'/weld_js_exe.csv', weld_js_exe, delimiter=',') # save welding/scanning logged joint space data
                        np.savetxt(logdata_dir+layer_name+f'/js_cmd.csv', q_cmd_all, delimiter=',') # save welding/scanning commanded joint space data
                        np.savetxt(logdata_dir+layer_name+f'/weld_cmd.csv', welding_cmd_all, delimiter=',') # save welding commands
                        if fuji_scanon:
                            with open(logdata_dir+layer_name+f'/scan_exe.pickle', 'wb') as file: # save scanning logged data
                                pickle.dump(scan_exe, file)
                        if thermal_on:
                            rr_sensors.stop_all_sensors() ## end thermal logging at the very end to collect more thermal data
                            rr_sensors.save_all_sensors(logdata_dir+layer_name+'/') # save thermal data
                    ##########################################
                else:
                    print("Read from file")
                    if weld_parts == 'base':
                        layer_name = 'baselayer'+str(i)
                    else:
                        layer_name = 'layer'+str(i)
                    weld_js_exe = np.loadtxt(logdata_dir+layer_name+f'/weld_js_exe.csv',delimiter=',')
                    with open(logdata_dir+layer_name+f'/scan_exe.pickle', 'rb') as f:
                        scan_exe = pickle.load(f)
                
                # assert len(stamps_exe) == len(scan_exe), f'Length of stamps_exe {len(stamps_exe)} and scan_exe {len(scan_exe)} do not match!'
                ################### get layer increments ############################
                if weld_arcon:
                    weld_js_exe = np.array(weld_js_exe)
                    stamps_exe = deepcopy(weld_js_exe[:,0])
                # if True:
                    # single scan noise remove
                    if not scan_online_process:
                        scan_exe_noise_remove = []
                        for scan in scan_exe:
                            scan_noise_remove = scan_process.scan2dDenoise(deepcopy(scan).T,crop_min=[-40,30],crop_max=[40,200])
                            scan_exe_noise_remove.append(scan_noise_remove)
                    if fuji_scanon:
                        with open(logdata_dir+layer_name+f'/scan_exe_noise_remove.pickle', 'wb') as f:
                            pickle.dump(scan_exe_noise_remove, f)
                    # 3D scan registration
                    pcd = scan_process.pcd_register_mti(scan_exe_noise_remove,weld_js_exe[:,np.append(np.arange(1,7),np.arange(13,15))],stamps_exe,flip=True,scanner='fuji')
                    # visualize_pcd([pcd])
                    curve_planned_z = np.mean(curve[:,2])
                    curve_x_end = np.min(curve[:,0])
                    curve_x_start = np.max(curve[:,0])
                    curve_y = np.mean(curve[:,1])
                    z_height_start=curve_planned_z+0.1
                    crop_extend_x=15
                    crop_extend_z=20
                    crop_min=(curve_x_end-crop_extend_x,curve_y-30,-30)
                    crop_max=(curve_x_start+crop_extend_x,curve_y+30,z_height_start+crop_extend_z)
                    crop_h_min=(curve_x_end-crop_extend_x,curve_y-20,-30)
                    crop_h_max=(curve_x_start+crop_extend_x,curve_y+20,z_height_start+crop_extend_z)
                    pcd = scan_process.pcd_noise_remove(pcd,outlier_remove=False,nb_neighbors=40,std_ratio=1.5,\
                                                        min_bound=crop_min,max_bound=crop_max,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=100)
                    # visualize_pcd([pcd])
                    # profile_height,Transz0_H = scan_process.pcd2height(deepcopy(pcd),z_height_start,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=Transz0_H)
                    profile_height,profile_width,Transz0_H = scan_process.pcd2height(deepcopy(pcd),z_height_start,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=Transz0_H,return_width=True)
                    print("Transz0_H:",Transz0_H)
                    np.savetxt(logdata_dir+layer_name+f'/profile_height.csv', profile_height, delimiter=',')
                    o3d.io.write_point_cloud(logdata_dir+layer_name+f'/pcd.pcd',pcd)

                    # update loglog-rls recursive least square
                    if control_method == 'loglog-rls' and weld_parts == 'layer' and layer_count >= correction_layer_start:
                        profile_height_shift = deepcopy(profile_height)
                        profile_width_shift = deepcopy(profile_width)
                        profile_height_shift[:,0] += shift_weld_profile_x
                        profile_width_shift[:,0] += shift_weld_profile_x
                        loglogModel.rls_update(profile_height_shift, last_profile_height, profile_width_shift, control_status_log)

                    if read_from_file_layer:
                        visualize_pcd([pcd])
                        plt.scatter(profile_height[:,0],profile_height[:,1])
                        plt.show()

                    if weld_parts == 'base':
                        mean_layer_height = np.mean(profile_height[:,1])
                    else:
                        # find the profile_height x>curve_x_start-shift_x and x<curve_x_end-shift_x
                        valid_indices = np.where((profile_height[:,0] > curve_x_end - shift_weld_profile_x) & (profile_height[:,0] < curve_x_start - shift_weld_profile_x))
                        mean_layer_height = np.mean(profile_height[valid_indices,1])
                    print("Mean Layer Height:",mean_layer_height)
                    last_profile_height = deepcopy(profile_height)
                    # with open(logdata_dir+layer_name+f'/profile_height.csv', 'wb') as f:
                    #     pickle.dump(profile_height, f)
                    
                    if weld_parts == 'base':
                        i = i+base_nom_incre # baselayer uses base_nom_incre
                    else:
                        i = round((mean_layer_height-2*baselayer_resolution)/layer_resolution) # layer uses mean_layer_height/layer_resolution
                else:
                    try:
                        # use a logged profile height as demo
                        profile_height = np.loadtxt(sim_folder+layer_name+f'/profile_height.csv',delimiter=',')
                        profile_width = np.loadtxt(sim_folder+layer_name+f'/profile_width.csv',delimiter=',')

                        # update loglog-rls recursive least square
                        if control_method == 'loglog-rls' and weld_parts == 'layer' and layer_count >= correction_layer_start:
                            profile_height_shift = deepcopy(profile_height)
                            profile_width_shift = deepcopy(profile_width)
                            profile_height_shift[:,0] += shift_weld_profile_x
                            profile_width_shift[:,0] += shift_weld_profile_x
                            # read weld cmd
                            weld_cmd = np.loadtxt(sim_folder+layer_name+f'/weld_cmd.csv',delimiter=',')
                            if weld_cmd.shape[1] < 5:
                                print("Weld command file does not have current x.")
                                js_cmd = np.loadtxt(sim_folder+layer_name+f'/js_cmd.csv',delimiter=',')
                                js_cmd_sample = []
                                for joint_i in range(14):
                                    js_cmd_sample.append(np.interp(weld_cmd[:,0], js_cmd[:,0], js_cmd[:,joint_i+2]))
                                js_cmd_sample = np.array(js_cmd_sample).T
                                cmd_x = []
                                for j_cmd in js_cmd_sample:
                                    t2 = positioner.fwd(j_cmd[-2:],world=True)
                                    t1 = robot_weld.fwd(j_cmd[:6],world=True)
                                    t1_t2 = t2.inv()*t1
                                    cmd_x.append(t1_t2.p[0])
                                weld_cmd = np.column_stack((weld_cmd[:,:1], cmd_x, weld_cmd[:,2:]))
                            weld_cmd = np.column_stack((weld_cmd, np.ones((weld_cmd.shape[0], 1)))) # add a dummy column for weld_cmd_updated
                            weld_cmd_full = []
                            for weld_cmd_i in range(len(weld_cmd)-1):
                                t_full = np.arange(weld_cmd[weld_cmd_i, 0], weld_cmd[weld_cmd_i+1, 0], 0.008)
                                x_full = np.interp(t_full, weld_cmd[weld_cmd_i:weld_cmd_i+2, 0], weld_cmd[weld_cmd_i:weld_cmd_i+2, 1])
                                torch_v_full = np.ones_like(x_full)*weld_cmd[weld_cmd_i, 2]
                                wire_feed_full = np.ones_like(x_full)*weld_cmd[weld_cmd_i, 3]
                                change_id_full = np.zeros_like(x_full)
                                change_id_full[0]=1 if weld_cmd_i > 0 else 0
                                weld_cmd_full.extend(np.column_stack((t_full, x_full, torch_v_full, wire_feed_full, change_id_full)))
                            weld_cmd_full = np.array(weld_cmd_full)

                            control_status_log = deepcopy(weld_cmd_full)

                            # plt.plot(control_status_log[:,1], control_status_log[:,2])
                            # plt.plot(control_status_log[:,1], control_status_log[:,-1])
                            # plt.show()

                            print("Use loglog RLS. Running RLS")
                            loglogModel.rls_update(profile_height_shift, last_profile_height, profile_width_shift, control_status_log)

                        if weld_parts == 'base':
                            mean_layer_height = np.mean(profile_height[:,1])
                            i = i+base_nom_incre
                        else:
                            # find the profile_height x>curve_x_start-shift_x and x<curve_x_end-shift_x
                            valid_indices = np.where((profile_height[:,0] > np.min(curve[:,0]) - shift_weld_profile_x) & (profile_height[:,0] < np.max(curve[:,0]) - shift_weld_profile_x))
                            mean_layer_height = np.mean(profile_height[valid_indices,1])
                            i = layer_nums[layer_count+1]
                        print("Mean Layer Height:",mean_layer_height)
                        last_profile_height = deepcopy(profile_height)
                    except FileNotFoundError:
                        if weld_parts == 'base':
                            i = i+nom_incre
                        else:
                            i = i+nom_incre
                ##########################################

                ### layer parameters update 
                layer_count += 1
                forward = not forward
                if read_from_file_layer:
                    input("Next layer "+str(i)+". Press Enter to continue...")
                read_from_file_layer = False
            except:
                traceback.print_exc()
                if weld_arcon:
                    fronius_client.stop_weld()
                    fronius_client.release_welder()
                SS.deinitialize_robot() if not SIMULATION else None
                if fuji_scanon and scan_online_process:
                    try:
                        while len(scan_process.raw_scan_pipe)!=0:
                            print("Final scan processing...",len(scan_process.raw_scan_pipe))
                            time.sleep(0.01)
                        while len(scan_process.denoise_pipe)!=0:
                            scan_denoise = scan_process.denoise_pipe.pop(0)
                            scan_exe_noise_remove.append(scan_denoise)
                        # stop scan process
                        scan_process.end_denoise_thread_flag = True
                        scan_denoise_thread.join()
                    except:
                        traceback.print_exc()
                break
    
    if weld_arcon:
        fronius_client.stop_weld()
        fronius_client.release_welder()
    SS.deinitialize_robot() if not SIMULATION else None

if __name__ == '__main__':
    main()
    