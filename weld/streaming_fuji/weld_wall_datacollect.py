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

def welding_profile_generate(lam_split, VPD, cross_section, layer_n, v_min, v_max):
    
    split_sections = len(lam_split)

    # random choose 1 from 2 cases
    case = np.random.randint(2)
    if case == 0: # monotonic increasing/decreasing
        v_start = np.random.uniform(v_min, v_max)
        v_end = np.random.uniform(v_min, v_max)
        vel_profile = np.linspace(v_start, v_end, split_sections)
    else: # wave-like
        v_start = np.random.uniform(v_min, v_max)
        v_amp = -1**(np.random.randint(2))*np.random.uniform((v_max-v_min)/10, (v_max-v_min)/2)
        vel_profile = v_start + v_amp*np.sin(np.linspace(0, 2*np.pi, split_sections))
        vel_profile = np.clip(vel_profile, v_min, v_max)
    # VPD = cross_section*inch2mm*layer_feedrate/layer_nom_vel # volume per distance (mm^3/mm)
    feedrate_profile = VPD*vel_profile*mm2inch/cross_section

    return vel_profile, feedrate_profile

def main():
    
    weld_arcon = False
    welder_log = False
    fuji_scanon = False
    scan_online_process = False
    thermal_on = False
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

    feedrate_update_rate=1.	#Hz
    job_offset=200
    # baselayer welding parameters
    base_feedrate = 250 
    base_nom_incre = 1
    base_nom_vel = 5
    # layer welding parameters
    layer_feedrate = 100 # inch/min
    layer_nom_height = 3 # mm
    layer_nom_vel = 4 # mm/s
    layer_nom_incre = int(layer_nom_height/layer_resolution)
    # weld starting point sleep
    weld_start_sleep = 0.2
    # scanning parameters
    scan_nom_vel = 5
    # collision avoidance z offset
    safety_z_offset = 50
    # direction 
    torch_ori_fix = False # torch orientation fixed
    
    # data collection parameters
    cross_section = 1.2 # mm^2
    VPD = cross_section*inch2mm*layer_feedrate/layer_nom_vel # volume per distance (mm^3/mm)
    split_sections = 6
    lam_split = np.linspace(0,meta_data['layer_length'],split_sections+1)[:-1]
    lam_split = lam_split.tolist()
    feedrate_min = 100
    feedrate_max = 220
    v_minimum = round(cross_section*inch2mm*feedrate_min/VPD,2)
    v_maximum = cross_section*inch2mm*feedrate_max/VPD
    print("VPD:",VPD)
    print("v_minimum:",v_minimum)
    print("v_maximum:",v_maximum)

    # start-end layers
    baselayer_start = 0
    baselayer_end = base_layer_num
    layer_start = 0
    layer_end = layer_num
    
    ################## Log data dir ##################
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-7]
    logdata_dir='../../data/wall_weld_test/weld_fujiscan_'+formatted_time+'/'

    read_from_file_layer = False
    Transz0_H=None
    if read_from_file_layer:
        logdata_dir = '../../data/wall_weld_test/weld_fujiscan_2025_03_03_18_10_13/'
        Transz0_H = [[ 9.99996717e-01, -9.38152707e-06,  2.56242820e-03, -1.69755313e-02],\
                    [-9.38152707e-06,  9.99973192e-01,  7.32226246e-03, -4.85084015e-02],\
                    [-2.56242820e-03, -7.32226246e-03 , 9.99969909e-01, -6.62458387e+00],\
                    [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00,  1.00000000e+00]]

    weld_meta_data = {'well_arcon':weld_arcon, 'fuji_scanon':fuji_scanon, 'data_dir':data_dir, 'logdata_dir':logdata_dir\
        ,'base_layer_num':base_layer_num, 'baselayer_resolution':baselayer_resolution, 'layer_num':layer_num, 'layer_resolution':layer_resolution\
        ,'base_feedrate':base_feedrate, 'base_nom_incre':base_nom_incre, 'base_nom_vel':base_nom_vel\
        , 'layer_feedrate':layer_feedrate, 'layer_nom_incre':layer_nom_incre, 'layer_nom_vel':layer_nom_vel\
        ,'corss_section':cross_section, 'VPD':VPD, 'split_sections':split_sections, 'lam_split':lam_split\
        ,'v_minimum':v_minimum, 'v_maximum':v_maximum, 'weld_start_sleep':weld_start_sleep}

    # get robot 2 resting pose
    q_cur = deepcopy(SS.q_cur)

    print("Logged Data Dir:",logdata_dir)
    input("Ready to start? Press Enter to continue...")
    ################## print layers ##################
    arc_off=True
    forward = True

    mean_layer_height = 0
    for weld_parts in ['base','layer']:
    # for weld_parts in ['layer']:
        if weld_parts == 'base':
            weld_start = baselayer_start
            weld_end = baselayer_end
            nom_incre = base_nom_incre
        else:
            weld_start = layer_start
            weld_end = layer_end
            nom_incre = layer_nom_incre
        layer_count = 0
        i=weld_start
        input("Start with layer "+str(i)+". Press Enter to continue...")
        while i < weld_end:
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
                
                if not read_from_file_layer:
                
                    # random generate current layer feedrate, velocity
                    if weld_parts == 'layer':
                        if layer_count<4:
                            vel_profile, feedrate_profile = welding_profile_generate(lam_split, VPD, cross_section, i, v_minimum, v_maximum*0.8)
                        else:
                            vel_profile, feedrate_profile = welding_profile_generate(lam_split, VPD, cross_section, i, v_minimum, v_maximum)
                        assert len(vel_profile) == len(feedrate_profile)
                        assert len(vel_profile) == len(lam_split), f'{len(vel_profile)} {len(feedrate_profile)} {len(lam_split)}'
                    else:
                        vel_profile = [base_nom_vel]*len(lam_split)
                        feedrate_profile = [base_feedrate]*len(lam_split)

                    ### information print
                    print(f'Velocity Profile: {vel_profile}')
                    print(f'Feedrate Profile: {feedrate_profile}')

                    if input_from_user:
                        input("Press Enter to continue...")
                    else:
                        # time.sleep(1)
                        pass

                    # move to start point with safety_z_offset
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
                    time.sleep(0.1)

                    input("Start")

                    # add a random delay
                    if layer_count < 99999999999:
                        wait_time = 0
                    else:
                        wait_time = np.random.uniform(0,10)
                    print("Wait for",wait_time,"s")
                    time.sleep(wait_time)

                    # start joints recording
                    ####### welding motion ##########################
                    lam_cur=0
                    # last_update_time=time.perf_counter()+5.
                    q_cmd_all = []
                    welding_cmd_all = []
                    weld_js_exe = []
                    scan_exe = []
                    if scan_online_process:
                        scan_exe_noise_remove = []
                        scan_denoise_thread = Thread(target=scan_process.scan_denoise_thread, args=([-40, 30],[40, 200])) # arges: (crop_min, crop_max)
                        scan_denoise_thread.start()
                    # initial velocity
                    v_cmd = vel_profile[0]
                    feedrate_cmd = feedrate_profile[0]
                    if thermal_on:
                        rr_sensors.start_all_sensors()
                    # time.sleep(3)
                    q_cur = deepcopy(SS.q_cur)
                    # motion_start_time = time.time()
                    # test_motion_start = True
                    while lam_cur<lam_relative[-1] - v_cmd/SS.streaming_rate:
                        loop_start=time.perf_counter()
                        # if test_motion_start and np.linalg.norm(q_cur-SS.q_cur)>1e-7:
                        #     print("Motion lag (start move):",time.time()-motion_start_time)
                        #     test_motion_start = False
                        q_cur = deepcopy(SS.q_cur)

                        ### get the next q commands
                        lam_cur+=v_cmd/SS.streaming_rate # get the current lambda (path location)
                        lam_idx=np.where(lam_relative>=lam_cur)[0][0] #get closest two indices and interpolate the joint angle
                        ratio=(lam_cur-lam_relative[lam_idx-1])/(lam_relative[lam_idx]-lam_relative[lam_idx-1])
                        q1=curve_js[lam_idx-1]*(1-ratio)+curve_js[lam_idx]*ratio
                        q2=curve_js_cam[lam_idx-1]*(1-ratio)+curve_js_cam[lam_idx]*ratio
                        q_pos=curve_js_positioner[lam_idx-1]*(1-ratio)+curve_js_positioner[lam_idx]*ratio
                        q_cmd=np.hstack((q1,q2,q_pos))

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

                        ###update welding param
                        if time.perf_counter()-last_update_time>1./feedrate_update_rate:
                            if weld_parts == 'layer':
                                # find the last index smaller than lam_cur
                                lam_idx=np.where(lam_split-v_cmd*feedrate_update_rate/2<=lam_cur)[0][-1]
                                v_cmd = vel_profile[lam_idx]
                                feedrate_cmd = feedrate_profile[lam_idx]
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
                            scan_process.raw_scan_pipe.append(deepcopy(line_profile))
                            while len(scan_process.denoise_pipe)!=0:
                                scan_denoise = scan_process.denoise_pipe.pop(0)
                                scan_exe_noise_remove.append(scan_denoise)

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

                    input("end")

                    ###### Motion varification
                    # time.sleep(1/SS.streaming_rate)
                    # motion_end_time = time.time()
                    # print_time = time.time()
                    # print("last q cur:",np.degrees(q_cur))
                    # print("q cur:",np.degrees(SS.q_cur))
                    # print("q cmd:",np.degrees(q_cmd))
                    # while np.linalg.norm(q_cur-SS.q_cur)>1e-7:
                    #     q_cur = deepcopy(SS.q_cur)
                    #     time.sleep(1/SS.streaming_rate)
                    #     if time.time()-print_time>1:
                    #         print("Motion lag")
                    #         print("last q cur:",np.degrees(q_cur))
                    #         print("q cur:",np.degrees(SS.q_cur))
                    #         print("q cmd:",np.degrees(q_cmd))
                    #         print_time = time.time()
                    # print("Motion lag Time:",time.time()-motion_end_time)
                    ######

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
                            valid_indices=np.intersect1d(valid_indices,np.where(np.abs(wire_packet[1].Z_data)>50)[0])
                            line_profile=np.hstack((wire_packet[1].Y_data[valid_indices].reshape(-1,1),wire_packet[1].Z_data[valid_indices].reshape(-1,1)))
                            scan_exe.append(line_profile)
                        weld_js_exe.append(np.append(time.perf_counter(),deepcopy(SS.q_cur))) # log robot joints

                        ### scan online processing
                        if fuji_scanon and scan_online_process:
                            scan_process.raw_scan_pipe.append(deepcopy(line_profile))
                            while len(scan_process.denoise_pipe)!=0:
                                scan_denoise = scan_process.denoise_pipe.pop(0)
                                scan_exe_noise_remove.append(scan_denoise)

                        ### sent position Command to the robot
                        q_cmd_all.append(np.hstack((time.perf_counter(),i,q_cmd)))
                        if lam_cur>lam_scan_relative[-1]-v_cmd/SS.streaming_rate:
                            SS.position_cmd(q_cmd)
                        else:
                            SS.position_cmd(q_cmd,loop_start)
                    ########################################
                    time.sleep(0.3) # for robot to drive to the end point

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
                    for z in np.arange(0,safety_z_offset+1,5): # a linear movement
                        T_end = robot_weld.fwd(curve_js_scan[-1])
                        T_end.p[2] += z
                        curve_js_end_offset = robot_weld.inv(T_end.p, T_end.R, last_joints=curve_js_scan[-1])[0]
                        q_end_offset = np.hstack((curve_js_end_offset, r2_rest_q, q_pos))
                        SS.jog2q(q_end_offset)

                    ############## save data ######################
                    if not os.path.exists(logdata_dir):
                        os.makedirs(logdata_dir)
                    # save meta data
                    with open(logdata_dir+'weld_meta_data.yml', 'w') as f:
                        yaml.dump(weld_meta_data, f)
                    if weld_parts == 'base':
                        layer_name = 'baselayer'+str(i)
                        pathlib.Path(logdata_dir+layer_name).mkdir(parents=True, exist_ok=True)
                    else:
                        layer_name = 'layer'+str(i)
                        pathlib.Path(logdata_dir+layer_name).mkdir(parents=True, exist_ok=True)
                    np.savetxt(logdata_dir+layer_name+f'/weld_js_exe.csv', weld_js_exe, delimiter=',') # save welding/scanning logged joint space data
                    np.savetxt(logdata_dir+layer_name+f'/js_cmd.csv', q_cmd_all, delimiter=',') # save welding/scanning commanded joint space data
                    np.savetxt(logdata_dir+layer_name+f'/weld_cmd.csv', welding_cmd_all, delimiter=',') # save welding commands
                    if fuji_scanon:
                        with open(logdata_dir+layer_name+f'/scan_exe.pickle', 'wb') as file: # save scanning logged data
                            pickle.dump(scan_exe, file)
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
                if weld_arcon:
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
                    curve_planned_z = np.mean(curve[:,2])
                    curve_x_end = np.min(curve[:,0])
                    curve_x_start = np.max(curve[:,0])
                    curve_y = np.mean(curve[:,1])
                    z_height_start=curve_planned_z+0.1
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
                    # with open(logdata_dir+layer_name+f'/profile_height.csv', 'wb') as f:
                    #     pickle.dump(profile_height, f)
                    np.savetxt(logdata_dir+layer_name+f'/profile_height.csv', profile_height, delimiter=',')
                    o3d.io.write_point_cloud(logdata_dir+layer_name+f'/pcd.pcd',pcd)
                    if weld_parts == 'base':
                        i = i+base_nom_incre # baselayer uses base_nom_incre
                    else:
                        i = round((mean_layer_height-2*baselayer_resolution)/layer_resolution) # layer uses mean_layer_height/layer_resolution
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
            except:
                traceback.print_exc()
                if weld_arcon:
                    fronius_client.stop_weld()
                    fronius_client.release_welder()
                SS.deinitialize_robot()
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
    SS.deinitialize_robot()

if __name__ == '__main__':
    main()
    