import time, os, copy, sys, yaml, pathlib
import traceback
from copy import deepcopy
import numpy as np
import datetime
from motoman_def import *
from lambda_calc import *
import open3d as o3d
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

inch2mm = 25.4
mm2inch = 1/25.4

def welder_handler(exp):
	if (exp is not None):
		# If "err" is not None it means that an exception occurred.
		# "err" contains the exception object
		print ("An error occured! " + str(exp))
		return

def main():
    
    weld_arcon = True
    welder_log = True
    fuji_scanon = True
    scan_online_process = True
    adaptive_layer_height = True
    compensate_shifting = True
    thermal_on = True
    input_from_user = False

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
    # RR_robot_sub = RRN.SubscribeService('rr+tcp://192.168.55.12:59945?service=robot')
    RR_robot_sub = RRN.SubscribeService('rr+tcp://localhost:59945?service=robot')
    point_distance=0.04		###STREAMING POINT INTERPOLATED DISTANCE
    SS=StreamingSend(RR_robot_sub,streaming_rate=125.)
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
    if welder_log and not weld_arcon:
        fronius_sub=RRN.SubscribeService('rr+tcp://192.168.55.21:60823?service=welder')
    
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
            rr_sensors = WeldRRSensor(weld_service=fronius_sub,cam_service=cam_ser)
        else:
            rr_sensors = WeldRRSensor(cam_service=cam_ser)
        # print("Test 3 Sec.")
        # rr_sensors.test_all_sensors()
        # print(len(rr_sensors.ir_recording))
        # rr_sensors.save_all_sensors('')
        # fronius_client.release_welder()
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

    # welder config
    feedrate_update_rate=1.	#Hz
    job_offset=200

    # target dh (dh start)
    target_dh = 2.3

    # baselayer welding parameters
    base_feedrate = 250 
    base_nom_incre = 1
    base_nom_vel = 5
    # layer welding parameters
    layer_feedrate = 100 # inch/min
    layer_nom_vel = 5 # mm/s
    layer_nom_height = v2dh_loglog(layer_nom_vel,layer_feedrate) # mm
    print("Layer Nominal Height:",layer_nom_height)
    layer_nom_incre = int(layer_nom_height/layer_resolution)
    # weld starting point sleep
    weld_start_sleep = 0.2
    # scanning parameters
    scan_nom_vel = 5
    # collision avoidance z offset
    safety_z_offset = 50
    # compensate for shifted weld
    shift_pos_smoother = 501
    # direction 
    # torch_ori_fix = False # torch orientation fixed
    correction_layer = 2
    offset_z = -7.4
    
    # data collection parameters
    cross_section = 1.2 # mm^2
    VPD = cross_section*inch2mm*layer_feedrate/layer_nom_vel # volume per distance (mm^3/mm)
    section_dlam = 2 ## mm
    split_sections = int(meta_data['layer_length']/section_dlam)
    lam_split = np.linspace(0,meta_data['layer_length'],split_sections+1)[:-1]
    feedrate_min = 100
    feedrate_max = 220
    # v_minimum = round(cross_section*inch2mm*feedrate_min/VPD,2)
    # v_maximum = cross_section*inch2mm*feedrate_max/VPD
    v_minimum = 1
    v_maximum = 12
    print("VPD:",VPD)
    print("v_minimum:",v_minimum)
    print("v_maximum:",v_maximum)

    # start-end layers
    baselayer_start = 0
    baselayer_end = base_layer_num
    layer_start = 9 # nominal baselayer=5.5. Real data=6.4 (6.4-5.5)/0.1=9
    layer_end = layer_num
    # layer_end = 10
    
    ################## Log data dir ##################
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-7]
    logdata_dir='../../data/wall_weld_test/weld_fujicontrol_'+formatted_time+'/'

    read_from_file_layer = False
    Transz0_H=None
    if read_from_file_layer:
        logdata_dir = '../../data/wall_weld_test/weld_fujiscan_2025_03_03_18_10_13/'
        Transz0_H = [[9.99996624e-01 ,-1.03837069e-05 , 2.59858273e-03, -1.92253817e-02],\
                    [-1.03837069e-05,  9.99968066e-01,  7.99168212e-03, -5.91257447e-02],\
                    [-2.59858273e-03 ,-7.99168212e-03 , 9.99964690e-01, -7.39814924e+00],\
                    [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]]

    weld_meta_data = {'well_arcon':weld_arcon, 'fuji_scanon':fuji_scanon, 'data_dir':data_dir, 'logdata_dir':logdata_dir\
        ,'base_layer_num':base_layer_num, 'baselayer_resolution':baselayer_resolution, 'layer_num':layer_num, 'layer_resolution':layer_resolution\
        ,'base_feedrate':base_feedrate, 'base_nom_incre':base_nom_incre, 'base_nom_vel':base_nom_vel\
        , 'layer_feedrate':layer_feedrate, 'layer_nom_incre':layer_nom_incre, 'layer_nom_vel':layer_nom_vel\
        ,'corss_section':cross_section, 'VPD':VPD, 'split_sections':split_sections, 'lam_split':lam_split.tolist()\
        ,'v_minimum':v_minimum, 'v_maximum':v_maximum, 'weld_start_sleep':weld_start_sleep\
        ,'target_dh':target_dh, 'correction_layer':correction_layer}

    # get robot 2 resting pose
    q_cur = deepcopy(SS.q_cur)

    print("Logged Data Dir:",logdata_dir)
    input("Ready to start? Press Enter to continue...")
    ################## print layers ##################
    last_layer_scan = False
    arc_off=True
    forward = True

    mean_layer_height = 0
    for weld_parts in ['base','layer']:
    # for weld_parts in ['layer']:
        if weld_parts == 'base':
            weld_start = baselayer_start
            weld_end = baselayer_end
            nom_incre = base_nom_incre
            nom_feedrate = base_feedrate
            nom_velocity = base_nom_vel
        else:
            weld_start = layer_start
            weld_end = layer_end
            nom_incre = layer_nom_incre
            nom_feedrate = layer_feedrate
            nom_velocity = layer_nom_vel
            
        layer_count = 0
        i=weld_start
        input("Start with layer "+str(i)+". Press Enter to continue...")
        while i < weld_end or last_layer_scan == False:
            if i>=weld_end and weld_parts=='base': # base layer don't need to last layer scan
                break
            if i >= weld_end:
                last_layer_scan = True
                weld_arcon = False
                i = weld_end - 1
                print("welding turn",weld_arcon)
                input("Last layer scannning. Press Enter to continue...")

            print("=====================================")
            print(f'Welding {weld_parts} layer {i} counting {layer_count} direction {forward}')
            try:
                if forward:
                    curve_direction = 'forward'
                else:
                    curve_direction = 'backward'
                
                # read curve joint space data
                if weld_parts == 'base':
                    curve = np.loadtxt(data_dir+f'curve_sliced_relative/baselayer{i}_0.csv',delimiter=',')
                    curve_scan = np.loadtxt(data_dir+f'curve_sliced_relative/baselayer{i}_0_scan_{curve_direction}.csv',delimiter=',')[::-1]
                    curve_js = np.loadtxt(data_dir+f'curve_sliced_js/MA2010_base_js{i}_0_{curve_direction}.csv', delimiter=',')[::-1]
                    curve_js_cam = np.loadtxt(data_dir+f'curve_sliced_js/MA1440_base_js{i}_0_{curve_direction}.csv', delimiter=',')[::-1]
                    curve_js_positioner = np.loadtxt(data_dir+f'curve_sliced_js/D500B_base_js{i}_0_{curve_direction}.csv', delimiter=',')[::-1]
                    curve_js_scan = np.loadtxt(data_dir+f'curve_sliced_js/MA2010_base_js{i}_0_scan_{curve_direction}.csv', delimiter=',')[::-1]
                    curve_js_pos_scan = np.loadtxt(data_dir+f'curve_sliced_js/D500B_base_js{i}_0_scan_{curve_direction}.csv', delimiter=',')[::-1]
                else:
                    curve = np.loadtxt(data_dir+f'curve_sliced_relative/slice{i}_0.csv',delimiter=',')
                    curve_scan = np.loadtxt(data_dir+f'curve_sliced_relative/slice{i}_0_scan_{curve_direction}.csv',delimiter=',')[::-1]
                    curve_js = np.loadtxt(data_dir+f'curve_sliced_js/MA2010_js{i}_0_{curve_direction}.csv', delimiter=',')[::-1]
                    curve_js_cam = np.loadtxt(data_dir+f'curve_sliced_js/MA1440_js{i}_0_{curve_direction}.csv', delimiter=',')[::-1]
                    curve_js_positioner = np.loadtxt(data_dir+f'curve_sliced_js/D500B_js{i}_0_{curve_direction}.csv', delimiter=',')[::-1]
                    curve_js_scan = np.loadtxt(data_dir+f'curve_sliced_js/MA2010_js{i}_0_scan_{curve_direction}.csv', delimiter=',')[::-1]
                    curve_js_pos_scan = np.loadtxt(data_dir+f'curve_sliced_js/D500B_js{i}_0_scan_{curve_direction}.csv', delimiter=',')[::-1]
                
                if forward: # since the scanner is leading, the welding direction is opposite to the planned curve direction
                    curve = curve[::-1]
                lam_relative = calc_lam_cs(curve[:,:3])
                lam_scan_relative = calc_lam_cs(curve_scan[:,:3])

                if not read_from_file_layer:
                    if input_from_user:
                        input("Press Enter to continue...")
                    else:
                        pass

                    # start with scanning 
                    # move to start point with safety_z_offset
                    r2_rest_q = curve_js_cam[0]
                    for z in np.arange(safety_z_offset,0,-5): # a linear movement
                        T_end = robot_weld.fwd(curve_js_scan[0])
                        T_end.p[2] += z
                        curve_js_end_offset = robot_weld.inv(T_end.p, T_end.R, last_joints=curve_js_scan[0])[0]
                        q_end_offset = np.hstack((curve_js_end_offset, r2_rest_q, curve_js_pos_scan[0]))
                        SS.jog2q(q_end_offset)
                    
                    # recording lam height and location
                    lam_state_height = []
                    for j in range(len(lam_split)):
                        lam_state_height.append([])
                    lam_curve_shift = np.array([[-1,0,0]]) # [lam, x, y]
                    # target p (only z matters)
                    target_p = deepcopy(curve[0][:3])
                    target_p[2] += target_dh

                    ####### scanning motion without welding ##########################
                    q_cmd_all = []
                    welding_cmd_all = []
                    weld_js_exe = []
                    scan_exe = []
                    if scan_online_process and fuji_scanon:
                        scan_exe_noise_remove = []
                        scan_exe_noise_remove_tcp = []
                        scan_dh_thread = Thread(target=scan_process.scan2dh_thread, args=(target_p,[-40, 30],[40, 200],offset_z,'fuji'),daemon=True) # arges: (target_p, crop_min, crop_max, offset_z, scanner)
                        scan_dh_thread.start()
                    v_cmd = scan_nom_vel
                    lam_cur=0
                    while lam_cur<lam_scan_relative[-1] - v_cmd/SS.streaming_rate:
                        loop_start=time.perf_counter()

                        ### get the next q commands
                        lam_cur+=v_cmd/SS.streaming_rate # get the current lambda (path location)
                        lam_idx=np.where(lam_scan_relative>=lam_cur)[0][0] #get closest two indices and interpolate the joint angle
                        ratio=(lam_cur-lam_scan_relative[lam_idx-1])/(lam_scan_relative[lam_idx]-lam_scan_relative[lam_idx-1])
                        q1=curve_js_scan[lam_idx-1]*(1-ratio)+curve_js_scan[lam_idx]*ratio
                        q_pos=curve_js_pos_scan[lam_idx-1]*(1-ratio)+curve_js_pos_scan[lam_idx]*ratio
                        q_cmd=np.hstack((q1,r2_rest_q,q_pos))

                        ### log data
                        if fuji_scanon:
                            wire_packet=fuji_scan_wire.TryGetInValue() # log fuji cam scanner data
                            valid_indices=np.where(wire_packet[1].I_data>1)[0]
                            valid_indices=np.intersect1d(valid_indices,np.where(np.abs(wire_packet[1].Z_data)>30)[0])
                            line_profile=np.hstack((wire_packet[1].Y_data[valid_indices].reshape(-1,1),wire_packet[1].Z_data[valid_indices].reshape(-1,1)))
                            scan_exe.append(line_profile)
                        weld_js_exe.append(np.append(time.perf_counter(),deepcopy(SS.q_cur))) # log robot joints

                        ### scan online processing
                        if fuji_scanon and scan_online_process:
                            while scan_process.accessing_key:
                                time.sleep(0.0000000000001)
                            scan_process.accessing_key = True
                            scan_process.raw_scan_pipe.append(deepcopy(line_profile))
                            scan_process.robot_q_pipe.append(deepcopy(weld_js_exe[-1][np.array([1,2,3,4,5,6,13,14])])) # log robot joints (robot 1 and positioner)
                            scan_process.accessing_key = False
                            while len(scan_process.denoise_scan_pipe)!=0:
                                while scan_process.accessing_key:
                                    time.sleep(0.0000000000001)
                                scan_process.accessing_key = True
                                scan_denoise = scan_process.denoise_scan_pipe.pop(0)
                                scan_denoise_tcp = scan_process.denoise_scan_tcp_pipe.pop(0)
                                scan_point_location = scan_process.point_location_pipe.pop(0)
                                scan_delta_h = scan_process.delta_h_pipe.pop(0)
                                scan_process.accessing_key = False
                                # get denoise scan
                                scan_exe_noise_remove.append(scan_denoise)
                                scan_exe_noise_remove_tcp.append(scan_denoise_tcp)
                                # get lambda and record height
                                curve_index = np.argsort(np.linalg.norm(curve[:,:2]-scan_point_location[:2],axis=1))[0]
                                lam_scan = lam_relative[curve_index]
                                lam_scan_i = np.where(lam_split<=lam_scan)[0][-1]
                                if scan_delta_h<10:
                                    lam_state_height[lam_scan_i].append(scan_delta_h)
                                curve_shift = scan_point_location[:2]-curve[curve_index][:2]
                                lam_curve_shift = np.vstack((lam_curve_shift,np.hstack((lam_scan,curve_shift))))

                        ### sent position Command to the robot
                        q_cmd_all.append(np.hstack((time.perf_counter(),i,q_cmd)))
                        if lam_cur>lam_scan_relative[-1]-v_cmd/SS.streaming_rate:
                            SS.position_cmd(q_cmd)
                        else:
                            SS.position_cmd(q_cmd,loop_start)
                    ########################################

                    # show height
                    for j, lam_height in enumerate(lam_state_height):
                        if len(lam_height) != 0:
                            print(f"Layer {i} Section {j} dh: {np.mean(lam_height)}")

                    lam_curve_shift = lam_curve_shift[1:]
                    lam_curve_shift = lam_curve_shift[lam_curve_shift[:,0]!=0]

                    print("Current mean shift x,y:",np.mean(lam_curve_shift[:,1]),np.mean(lam_curve_shift[:,2]))
                    print("Current std shift x,y:",np.std(lam_curve_shift[:,1]),np.std(lam_curve_shift[:,2]))
                    lam_curve_shift = lam_curve_shift[np.argsort(lam_curve_shift[:,0])]
                    # 1d smoother
                    lam_curve_shift_noise = np.where(np.abs(lam_curve_shift[:,2]-np.mean(lam_curve_shift[:,2]))>3*np.std(lam_curve_shift[:,2]))[0]
                    print("Noise index:",lam_curve_shift_noise)
                    lam_curve_shift[lam_curve_shift_noise,2] = np.mean(lam_curve_shift[:,2])
                    lam_curve_shift_smooth_x = moving_average(lam_curve_shift[:,1],n=shift_pos_smoother,padding=True)
                    lam_curve_shift_smooth_y = moving_average(lam_curve_shift[:,2],n=shift_pos_smoother,padding=True)
                    # plt.plot(lam_curve_shift[:,1])
                    # plt.plot(lam_curve_shift_smooth_x)
                    # plt.show()
                    # plt.plot(lam_curve_shift[:,2])
                    # plt.plot(lam_curve_shift_smooth_y)
                    # plt.show()
                    # lam_curve_shift[:,1] = lam_curve_shift_smooth_x
                    # lam_curve_shift[:,2] = lam_curve_shift_smooth_y
                    if compensate_shifting and not (weld_parts == 'base' and layer_count==0):
                        # get the robot to the shifted position
                        shifted_xy = np.mean(lam_curve_shift[:shift_pos_smoother+1,1:],axis=0)
                        print("Shifted x,y:",shifted_xy)
                        shifted_xy = np.array([0,0])
                        if not forward:
                            shifted_xy = np.array([0, -0.8304026453085149])
                        T_positioner_world = positioner.fwd(curve_js_positioner[0],world=True)
                        T_robot_origin = robot_weld.fwd(curve_js[0])
                        T_robot_positioner = T_positioner_world.inv()*T_robot_origin
                        T_robot_shift = T_robot_positioner
                        T_robot_shift.p[:2] += shifted_xy
                        T_robot_shift = T_positioner_world*T_robot_shift
                        q_shift = robot_weld.inv(T_robot_shift.p, T_robot_shift.R, last_joints=curve_js[0])[0]
                        q_cmd[:6] = q_shift # only update the robot 1 joints
                        time.sleep(0.1)
                        SS.jog2q(q_cmd)

                    time.sleep(0.5) # for robot to drive to the end point

                    ####### welding motion ##########################
                    lam_cur=0
                    lam_split_i = 0
                    # initial velocity
                    if weld_parts == 'base' or layer_count<correction_layer:
                        v_cmd = nom_velocity
                    else:
                        v_cmd = dh2v_loglog(np.mean(lam_state_height[0]),mode=nom_feedrate)
                        v_cmd = np.clip(v_cmd,v_minimum,v_maximum)
                    feedrate_cmd = nom_feedrate #TODO: update feedrate based on the height, width, thermal
                    if thermal_on:
                        rr_sensors.start_all_sensors()
                    while lam_cur<lam_relative[-1] - v_cmd/SS.streaming_rate:
                        loop_start=time.perf_counter()

                        ### get the next q commands
                        lam_cur+=v_cmd/SS.streaming_rate # get the current lambda (path location)
                        lam_idx=np.where(lam_relative>=lam_cur)[0][0] #get closest two indices and interpolate the joint angle
                        ratio=(lam_cur-lam_relative[lam_idx-1])/(lam_relative[lam_idx]-lam_relative[lam_idx-1])
                        q1=curve_js[lam_idx-1]*(1-ratio)+curve_js[lam_idx]*ratio
                        q2=curve_js_cam[lam_idx-1]*(1-ratio)+curve_js_cam[lam_idx]*ratio
                        q_pos=curve_js_positioner[lam_idx-1]*(1-ratio)+curve_js_positioner[lam_idx]*ratio
                        q_cmd=np.hstack((q1,q2,q_pos))

                        if compensate_shifting and not (weld_parts == 'base' and layer_count==0):
                            # get the robot to the shifted position
                            shifted_i = np.where(lam_curve_shift[:,0]>=lam_cur)[0][0]-1
                            ### moving average for shifting
                            if shifted_i < (shift_pos_smoother-1)/2:
                                shifted_xy = np.mean(lam_curve_shift[:shift_pos_smoother+1,1:3],axis=0)
                            elif shifted_i > len(lam_curve_shift)-(shift_pos_smoother-1)/2:
                                shifted_xy = np.mean(lam_curve_shift[-shift_pos_smoother:,1:3],axis=0)
                            else:
                                shifted_xy = np.mean(lam_curve_shift[shifted_i-int((shift_pos_smoother-1)/2):shifted_i+int((shift_pos_smoother-1)/2)+1,1:3],axis=0)
                            shifted_xy = np.array([0,0])
                            if not forward:
                                shifted_xy = np.array([0, -0.8304026453085149])
                            
                            # compensation in the positioner tcp frame
                            T_positioner_world = positioner.fwd(q_pos,world=True)
                            T_robot_origin = robot_weld.fwd(q1)
                            T_robot_positioner = T_positioner_world.inv()*T_robot_origin
                            T_robot_shift = T_robot_positioner
                            T_robot_shift.p[:2] += shifted_xy
                            # transfer back to the welding robot motion
                            T_robot_shift = T_positioner_world*T_robot_shift
                            q_shift = robot_weld.inv(T_robot_shift.p, T_robot_shift.R, last_joints=q1)[0]
                            q_cmd[:6] = q_shift # only update the robot 1 joints

                        ### if welding start or end
                        if arc_off:
                            if weld_arcon:
                                print("Welding Start")
                                fronius_client.job_number = int(round(feedrate_cmd/10)+job_offset)
                                fronius_client.start_weld()
                                time.sleep(weld_start_sleep)
                            welding_cmd_all.append(np.hstack((time.perf_counter(),i,v_cmd,int(round(feedrate_cmd/10)*10))))
                            last_update_time=time.perf_counter()
                            arc_off=False
                        
                        ### update speed 
                        if weld_parts != 'base' and layer_count>=correction_layer: 
                            if lam_split_i < len(lam_split)-1 and lam_cur > lam_split[lam_split_i+1]:
                                lam_split_i += 1
                                if len(lam_state_height[lam_split_i]) == 0 or np.mean(lam_state_height[lam_split_i]) < 0:
                                    lam_state_height[lam_split_i] = []
                                    lam_split_i_prev = lam_split_i
                                    while len(lam_state_height[lam_split_i_prev]) == 0:
                                        lam_split_i_prev -= 1
                                        if lam_split_i_prev < 0:
                                            break
                                        if len(lam_state_height[lam_split_i_prev]) != 0:
                                            lam_state_height[lam_split_i].append(np.mean(lam_state_height[lam_split_i_prev]))
                                            break
                                    lam_split_i_next = lam_split_i
                                    while len(lam_state_height[lam_split_i_next]) == 0:
                                        lam_split_i_next += 1
                                        if lam_split_i_next >= len(lam_split):
                                            break
                                        if len(lam_state_height[lam_split_i_next]) != 0:
                                            lam_state_height[lam_split_i].append(np.mean(lam_state_height[lam_split_i_next]))
                                            break
                                v_cmd_new = dh2v_loglog(np.mean(lam_state_height[lam_split_i]),mode=nom_feedrate)
                                if not np.isnan(v_cmd_new):
                                    v_cmd = v_cmd_new
                                v_cmd = np.clip(v_cmd,v_minimum,v_maximum)
                                welding_cmd_all.append(np.hstack((time.perf_counter(),i,v_cmd,int(round(feedrate_cmd/10)*10))))
                                print("Update Velocity:",round(v_cmd,2), "Mean dh:",np.mean(lam_state_height[lam_split_i]))

                        ###update welding param
                        if time.perf_counter()-last_update_time>1./feedrate_update_rate:
                            if weld_parts == 'layer':
                                # find the last index smaller than lam_cur
                                lam_idx=np.where(lam_split-v_cmd*feedrate_update_rate/2<=lam_cur)[0][-1]
                                v_cmd = v_cmd
                                feedrate_cmd = nom_feedrate #TODO: update feedrate based on the height, width, thermal
                                # update feedrate to welder
                                if weld_arcon:
                                    fronius_client.async_set_job_number(int(round(feedrate_cmd/10)+job_offset), welder_handler)
                            # log command data
                            welding_cmd_all.append(np.hstack((time.perf_counter(),i,v_cmd,int(round(feedrate_cmd/10)*10))))
                            last_update_time=time.perf_counter()
                            print("Update Feedrate, Velocity:",int(round(feedrate_cmd/10)*10),round(v_cmd,1))
                        
                        ### log data
                        if fuji_scanon:
                            wire_packet=fuji_scan_wire.TryGetInValue() # log fuji cam scanner data
                            valid_indices=np.where(wire_packet[1].I_data>1)[0]
                            valid_indices=np.intersect1d(valid_indices,np.where(np.abs(wire_packet[1].Z_data)>50)[0])
                            line_profile=np.hstack((wire_packet[1].Y_data[valid_indices].reshape(-1,1),wire_packet[1].Z_data[valid_indices].reshape(-1,1)))
                            scan_exe.append(line_profile)
                        weld_js_exe.append(np.append(time.perf_counter(),deepcopy(SS.q_cur))) # log timestamp and robot joints

                        ### scan online processing
                        if fuji_scanon and scan_online_process:
                            while scan_process.accessing_key:
                                time.sleep(0.0000000000001)
                            scan_process.accessing_key = True
                            scan_process.raw_scan_pipe.append(deepcopy(line_profile))
                            scan_process.robot_q_pipe.append(deepcopy(weld_js_exe[-1][np.array([1,2,3,4,5,6,13,14])])) # log robot joints (robot 1 and positioner)
                            scan_process.accessing_key = False
                            while len(scan_process.denoise_scan_pipe)!=0:
                                while scan_process.accessing_key:
                                    time.sleep(0.0000000000001)
                                scan_process.accessing_key = True
                                scan_denoise = scan_process.denoise_scan_pipe.pop(0)
                                scan_denoise_tcp = scan_process.denoise_scan_tcp_pipe.pop(0)
                                scan_point_location = scan_process.point_location_pipe.pop(0)
                                scan_delta_h = scan_process.delta_h_pipe.pop(0)
                                scan_process.accessing_key = False
                                # get denoise scan
                                scan_exe_noise_remove.append(scan_denoise)
                                scan_exe_noise_remove_tcp.append(scan_denoise_tcp)
                                # get lambda and record height
                                if lam_cur<lam_relative[int(-dist_weld_scan_index + 1/path_dl)]:
                                    curve_index = np.argsort(np.linalg.norm(curve[:,:2]-scan_point_location[:2],axis=1))[0]
                                    lam_scan = lam_relative[curve_index]
                                    lam_scan_i = np.where(lam_split<=lam_scan)[0][-1]
                                    lam_state_height[lam_scan_i].append(scan_delta_h)
                                    curve_shift = scan_point_location[:2]-curve[curve_index][:2]
                                    if np.abs(curve_shift[1]-np.mean(lam_curve_shift[:,2]))>2*np.std(lam_curve_shift[:,2]):
                                        curve_shift[1] = np.mean(lam_curve_shift[:,2])
                                    lam_curve_shift = np.vstack((lam_curve_shift,np.hstack((lam_scan,curve_shift))))

                        ### sent position Command to the robot
                        q_cmd_all.append(np.hstack((time.perf_counter(),i,q_cmd)))
                        if lam_cur>lam_relative[-1]-v_cmd/SS.streaming_rate:
                            SS.position_cmd(q_cmd,loop_start)
                        else:
                            SS.position_cmd(q_cmd,loop_start)

                    ### welding end
                    if weld_arcon:
                        fronius_client.stop_weld()
                        time.sleep(0.5)
                    arc_off=True
                    if thermal_on:
                        rr_sensors.stop_all_sensors()
                    ########################################

                    # final scan processing
                    if fuji_scanon and scan_online_process:
                        while len(scan_process.raw_scan_pipe)!=0:
                            print("Final scan processing...",len(scan_process.raw_scan_pipe))
                            time.sleep(0.01)
                        while len(scan_process.denoise_scan_pipe)!=0:
                            scan_denoise = scan_process.denoise_scan_pipe.pop(0)
                            scan_denoise_tcp = scan_process.denoise_scan_tcp_pipe.pop(0)
                            scan_point_location = scan_process.point_location_pipe.pop(0)
                            scan_delta_h = scan_process.delta_h_pipe.pop(0)
                            # get denoise scan
                            scan_exe_noise_remove.append(scan_denoise)
                            scan_exe_noise_remove_tcp.append(scan_denoise_tcp)
                            # get lambda and record height
                            # curve_index = np.argsort(np.linalg.norm(curve[:,:2]-scan_point_location[:2],axis=1))[0]
                            # lam_scan = lam_relative[curve_index]
                            # lam_scan_i = np.where(lam_split<=lam_scan)[0][-1]
                            # lam_state_height[lam_scan_i].append(scan_delta_h)
                            # curve_shift = scan_point_location[:2]-curve[curve_index][:2]
                            # lam_curve_shift = np.vstack((lam_curve_shift,np.hstack((lam_scan,curve_shift))))
                        # stop scan process
                        scan_process.end_denoise_thread_flag = True
                        scan_dh_thread.join()
                    
                    # show height
                    for j, lam_height in enumerate(lam_state_height):
                        if len(lam_height) != 0:
                            print(f"Layer {i} Section {j} dh: {np.mean(lam_height)}")

                    # move to end point with safety_z_offset
                    for z in np.arange(0,safety_z_offset+1,5): # a linear movement
                        T_end = robot_weld.fwd(q_cmd[:6])
                        T_end.p[2] += z
                        curve_js_end_offset = robot_weld.inv(T_end.p, T_end.R, last_joints=q_cmd[:6])[0]
                        q_end_offset = np.hstack((curve_js_end_offset, q_cmd[6:]))
                        SS.jog2q(q_end_offset)
                    time.sleep(0.1)

                    ############## save data ######################
                    if not os.path.exists(logdata_dir):
                        os.makedirs(logdata_dir)
                    # save meta data
                    with open(logdata_dir+'weld_meta_data.yml', 'w') as f:
                        yaml.dump(weld_meta_data, f)
                    if weld_parts == 'base':
                        layer_name = 'baselayer'+str(i)
                    elif last_layer_scan:
                        layer_name = 'layer'+str(i)+'_last'
                    else:
                        layer_name = 'layer'+str(i)
                    pathlib.Path(logdata_dir+layer_name).mkdir(parents=True, exist_ok=True)
                    np.savetxt(logdata_dir+layer_name+f'/weld_js_exe.csv', weld_js_exe, delimiter=',') # save welding/scanning logged joint space data
                    np.savetxt(logdata_dir+layer_name+f'/js_cmd.csv', q_cmd_all, delimiter=',') # save welding/scanning commanded joint space data
                    np.savetxt(logdata_dir+layer_name+f'/weld_cmd.csv', welding_cmd_all, delimiter=',') # save welding commands
                    if fuji_scanon:
                        with open(logdata_dir+layer_name+f'/scan_exe.pickle', 'wb') as file: # save scanning logged data
                            pickle.dump(scan_exe, file)
                        if scan_online_process:
                            with open(logdata_dir+layer_name+f'/scan_exe_noise_remove.pickle', 'wb') as f:
                                pickle.dump(scan_exe_noise_remove, f)
                    if thermal_on:
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
                
                weld_js_exe = np.array(weld_js_exe)
                stamps_exe = deepcopy(weld_js_exe[:,0])
                ################### get layer increments ############################
                # if weld_arcon and adaptive_layer_height:
                if adaptive_layer_height:
                    # single scan noise remove
                    if not scan_online_process:
                        scan_exe_noise_remove = []
                        for scan in scan_exe:
                            scan_noise_remove = scan_process.scan2dDenoise(deepcopy(scan).T,crop_min=[-40,30],crop_max=[40,200])
                            scan_exe_noise_remove.append(scan_noise_remove)
                        with open(logdata_dir+layer_name+f'/scan_exe_noise_remove.pickle', 'wb') as f:
                            pickle.dump(scan_exe_noise_remove, f)
                        # 3D scan registration
                        pcd = scan_process.pcd_register_mti(scan_exe_noise_remove,weld_js_exe[:,np.append(np.arange(1,7),np.arange(13,15))],stamps_exe,flip=True,scanner='fuji')
                    else:
                        pcd = o3d.geometry.PointCloud()
                        for scan_tcp in scan_exe_noise_remove_tcp:
                            pcd_slice = o3d.geometry.PointCloud()
                            pcd_slice.points=o3d.utility.Vector3dVector(scan_tcp)
                            pcd_slice = pcd_slice.voxel_down_sample(voxel_size=0.05)
                            pcd += pcd_slice
                        # visualize_pcd([pcd])
                    curve_planned_z = np.mean(curve[:,2])
                    curve_x_end = np.min(curve[:,0])
                    curve_x_start = np.max(curve[:,0])
                    curve_y = np.mean(curve[:,1])
                    z_height_start=curve_planned_z-5
                    crop_extend_x=10
                    crop_extend_z=20
                    crop_min=(curve_x_end-crop_extend_x,curve_y-30,-30)
                    crop_max=(curve_x_start+crop_extend_x,curve_y+30,z_height_start+crop_extend_z)
                    crop_h_min=(curve_x_end-crop_extend_x,curve_y-20,-30)
                    crop_h_max=(curve_x_start+crop_extend_x,curve_y+20,z_height_start+crop_extend_z)
                    pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
                                                        min_bound=crop_min,max_bound=crop_max,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=100)
                    profile_height,Transz0_H = scan_process.pcd2height(deepcopy(pcd),z_height_start,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=Transz0_H)
                    print("Transz0_H:",Transz0_H)
                    if read_from_file_layer:
                        visualize_pcd([pcd])
                        plt.scatter(profile_height[:,0],profile_height[:,1])
                        plt.show()

                    mean_layer_height = np.mean(profile_height[:,1])
                    print("Mean Layer Height:",mean_layer_height)
                    with open(logdata_dir+layer_name+f'/profile_height.csv', 'wb') as f:
                        pickle.dump(profile_height, f)
                    o3d.io.write_point_cloud(logdata_dir+layer_name+f'/pcd.pcd',pcd)
                    if weld_parts == 'base':
                        i = i+base_nom_incre # baselayer uses base_nom_incre
                    else:
                        # layer uses mean_layer_height/layer_resolution, 
                        # and add layer_nom_incre because scanner leads the welding
                        i = round((mean_layer_height-2*baselayer_resolution)/layer_resolution)+layer_nom_incre
                        print("Layer Incre to:",i)
                else:
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

                print("To next layer",i, last_layer_scan)
            except:
                traceback.print_exc()
                if weld_arcon:
                    fronius_client.stop_weld()
                    fronius_client.release_welder()
                SS.deinitialize_robot()
                break
    
    if weld_arcon:
        fronius_client.stop_weld()
        fronius_client.release_welder()
    
    SS.deinitialize_robot()

if __name__ == '__main__':
    main()
    