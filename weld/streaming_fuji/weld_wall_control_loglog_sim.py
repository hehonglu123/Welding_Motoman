import time, os, copy, sys, yaml, pathlib, glob
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
streaming_rate = 125

def welder_handler(exp):
	if (exp is not None):
		# If "err" is not None it means that an exception occurred.
		# "err" contains the exception object
		print ("An error occured! " + str(exp))
		return

def main():
    
    scan_online_process = True
    adaptive_layer_height = True

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

    scan_process = ScanProcess(robot_scan,positioner) # initialize scan process
    
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
    cross_section = 1.2 # mm^2
    feedrate_update_rate=1.	#Hz
    job_offset=200

    logdata_dir='../../data/wall_weld_test/weld_fujicontrol_2025_03_12_16_14_17/'
    # read weld meta data
    with open(logdata_dir+'weld_meta_data.yml', 'r') as f:
        weld_meta_data = yaml.safe_load(f)

    # target dh (dh start)
    target_dh = weld_meta_data['target_dh']
    # baselayer welding parameters
    base_feedrate = weld_meta_data['base_feedrate']
    base_nom_incre = weld_meta_data['base_nom_incre']
    base_nom_vel = weld_meta_data['base_nom_vel']
    # layer welding parameters
    layer_feedrate = weld_meta_data['layer_feedrate'] # inch/min
    layer_nom_vel = weld_meta_data['layer_nom_vel'] # mm/s
    layer_nom_height = v2dh_loglog(layer_nom_vel,layer_feedrate) # mm
    print("Layer Nominal Height:",layer_nom_height)
    layer_nom_incre = int(layer_nom_height/layer_resolution)
    # weld starting point sleep
    weld_start_sleep = weld_meta_data['weld_start_sleep']
    # compensate for shifted weld
    shift_pos_smoother = 61
    # direction 
    # torch_ori_fix = False # torch orientation fixed
    correction_layer = weld_meta_data['correction_layer']
    # offset_z
    offset_z = -7.4
    
    # data collection parameters
    VPD = cross_section*inch2mm*layer_feedrate/layer_nom_vel # volume per distance (mm^3/mm)
    section_dlam = 2 ## mm
    split_sections = int(meta_data['layer_length']/section_dlam)
    lam_split = np.linspace(0,meta_data['layer_length'],split_sections+1)[:-1]
    lam_split = lam_split.tolist()
    feedrate_min = 100
    feedrate_max = 220
    # v_minimum = round(cross_section*inch2mm*feedrate_min/VPD,2)
    # v_maximum = cross_section*inch2mm*feedrate_max/VPD
    v_minimum = weld_meta_data['v_minimum']
    v_maximum = weld_meta_data['v_maximum']
    print("VPD:",VPD)
    print("v_minimum:",v_minimum)
    print("v_maximum:",v_maximum)

    print("Logged Data Dir:",logdata_dir)
    input("Ready to start? Press Enter to continue...")
    ################## print layers ##################
    last_layer_scan = False
    arc_off=True
    forward = False

    Transz0_H=None
    mean_layer_height = 0
    # for weld_parts in ['base','layer']:
    for weld_parts in ['layer']:
        if weld_parts == 'base':
            total_layers_name = glob.glob(logdata_dir+'baselayer*')
        else:
            total_layers_name = glob.glob(logdata_dir+'layer*')
        # get printed layer number
        layer_nums = []
        for layer_name in total_layers_name:
            this_layer = layer_name.split('\\')[-1]
            this_layer = this_layer.split('r')[-1]
            layer_nums.append(int(this_layer))
        layer_nums = np.sort(layer_nums)
            
        
        
        for i in layer_nums[-5:]:
            layer_count = np.where(layer_nums==i)[0][0]
            forward = True if layer_count%2==0 else False

            print("=====================================")
            input(f'Welding {weld_parts} layer {i} counting {layer_count} direction {forward}')
            try:
                if forward:
                    curve_direction = 'forward'
                else:
                    curve_direction = 'backward'
                
                # read curve joint space data
                if weld_parts == 'base':
                    curve = np.loadtxt(data_dir+f'curve_sliced_relative/baselayer{i}_0.csv',delimiter=',')
                    # curve_scan = np.loadtxt(data_dir+f'curve_sliced_relative/baselayer{i}_0_scan_{curve_direction}.csv',delimiter=',')[::-1]
                    # curve_js = np.loadtxt(data_dir+f'curve_sliced_js/MA2010_base_js{i}_0_{curve_direction}.csv', delimiter=',')[::-1]
                    # curve_js_cam = np.loadtxt(data_dir+f'curve_sliced_js/MA1440_base_js{i}_0_{curve_direction}.csv', delimiter=',')[::-1]
                    # curve_js_positioner = np.loadtxt(data_dir+f'curve_sliced_js/D500B_base_js{i}_0_{curve_direction}.csv', delimiter=',')[::-1]
                    # curve_js_scan = np.loadtxt(data_dir+f'curve_sliced_js/MA2010_base_js{i}_0_scan_{curve_direction}.csv', delimiter=',')[::-1]
                    # curve_js_pos_scan = np.loadtxt(data_dir+f'curve_sliced_js/D500B_base_js{i}_0_scan_{curve_direction}.csv', delimiter=',')[::-1]
                else:
                    curve = np.loadtxt(data_dir+f'curve_sliced_relative/slice{i}_0.csv',delimiter=',')
                    # curve_scan = np.loadtxt(data_dir+f'curve_sliced_relative/slice{i}_0_scan_{curve_direction}.csv',delimiter=',')[::-1]
                    # curve_js = np.loadtxt(data_dir+f'curve_sliced_js/MA2010_js{i}_0_{curve_direction}.csv', delimiter=',')[::-1]
                    # curve_js_cam = np.loadtxt(data_dir+f'curve_sliced_js/MA1440_js{i}_0_{curve_direction}.csv', delimiter=',')[::-1]
                    # curve_js_positioner = np.loadtxt(data_dir+f'curve_sliced_js/D500B_js{i}_0_{curve_direction}.csv', delimiter=',')[::-1]
                    # curve_js_scan = np.loadtxt(data_dir+f'curve_sliced_js/MA2010_js{i}_0_scan_{curve_direction}.csv', delimiter=',')[::-1]
                    # curve_js_pos_scan = np.loadtxt(data_dir+f'curve_sliced_js/D500B_js{i}_0_scan_{curve_direction}.csv', delimiter=',')[::-1]
                
                if forward: # since the scanner is leading, the welding direction is opposite to the planned curve direction
                    curve = curve[::-1]
                lam_relative = calc_lam_cs(curve[:,:3])
                # lam_scan_relative = calc_lam_cs(curve_scan[:,:3])

                # r2 scanning steady pose
                # r2_rest_q = curve_js_cam[0]
                
                # recording lam height and location
                lam_state_height = []
                for j in range(len(lam_split)):
                    lam_state_height.append([])
                lam_curve_shift = np.array([[-1,0,0]]) # [lam, x, y]
                # target p (only z matters)
                target_p = deepcopy(curve[0][:3])
                target_p[2] += target_dh

                ####### scanning motion without welding ##########################
                ### read log data for simulation
                q_cmd_all = np.loadtxt(logdata_dir+f'layer{i}/js_cmd.csv',delimiter=',')
                welding_cmd_all = np.loadtxt(logdata_dir+f'layer{i}/weld_cmd.csv',delimiter=',')
                weld_js_exe = np.loadtxt(logdata_dir+f'layer{i}/weld_js_exe.csv',delimiter=',')
                with open(logdata_dir+f'layer{i}/scan_exe.pickle', 'rb') as file:
                    scan_exe = pickle.load(file)
                scan_exe_noise_remove = []
                scan_exe_noise_remove_tcp = []
                assert len(q_cmd_all)==len(weld_js_exe), "Command and execution data length mismatch"
                assert len(weld_js_exe)==len(scan_exe), "Execution and scanning data length mismatch"
                # find the biggest timestamp gap in weld_js_exe
                time_diff = np.diff(weld_js_exe[:,0])
                weld_start_index = np.argmax(time_diff)+1
                ###

                ### for visualization
                fig, ax = plt.subplots()
                ax.set_xlim(-20, 10)  # Set x-axis limits
                ax.set_ylim(100, 130)  # Set y-axis limits
                # Initialize an empty line object
                line, = ax.plot([], [], '.')  # No data at start
                line_large, = ax.plot([], [], '.', ms=10)  # No data at start
                plt.ion()  # Turn on interactive mode
                plt.show()
                ######
                
                scan_dh_thread = Thread(target=scan_process.scan2dh_thread, args=(target_p,[-40, 30],[40, 200],offset_z,'fuji'),daemon=True) # arges: (target_p, crop_min, crop_max, offset_z, scanner)
                scan_dh_thread.start()
                lam_cur=0
                for data_i in range(0,weld_start_index,100):
                    loop_start=time.perf_counter()

                    ### get the next q commands
                    # lam_cur+=v_cmd/streaming_rate # get the current lambda (path location)
                    # lam_idx=np.where(lam_scan_relative>=lam_cur)[0][0] #get closest two indices and interpolate the joint angle
                    # ratio=(lam_cur-lam_scan_relative[lam_idx-1])/(lam_scan_relative[lam_idx]-lam_scan_relative[lam_idx-1])
                    # q1=curve_js_scan[lam_idx-1]*(1-ratio)+curve_js_scan[lam_idx]*ratio
                    # q_pos=curve_js_pos_scan[lam_idx-1]*(1-ratio)+curve_js_pos_scan[lam_idx]*ratio
                    # q_cmd=np.hstack((q1,r2_rest_q,q_pos))

                    ### scan online processing
                    while scan_process.accessing_key:
                        time.sleep(0.0000000000001)
                    scan_process.accessing_key = True
                    scan_process.raw_scan_pipe.append(deepcopy(scan_exe[data_i]))
                    scan_process.robot_q_pipe.append(deepcopy(weld_js_exe[data_i][np.array([1,2,3,4,5,6,13,14])])) # log robot joints (robot 1 and positioner)
                    scan_process.accessing_key = False
                    while len(scan_process.denoise_scan_pipe)==0:
                        time.sleep(0.0000000000001) # wait for processing in simulation
                    while len(scan_process.denoise_scan_pipe)!=0:
                        while scan_process.accessing_key:
                            time.sleep(0.0000000000001)
                        scan_process.accessing_key = True
                        scan_denoise = scan_process.denoise_scan_pipe.pop(0)
                        scan_denoise_tcp = scan_process.denoise_scan_tcp_pipe.pop(0)
                        scan_point_location = scan_process.point_location_pipe.pop(0)
                        scan_delta_h = scan_process.delta_h_pipe.pop(0)
                        scan_process.accessing_key = False

                        line.set_data(scan_denoise_tcp[:,1], scan_denoise_tcp[:,2])  # Update both x and y
                        line_large.set_data([scan_point_location[1]], [scan_point_location[2]-offset_z])  # Update both x and y
                        ax.set_xlim(30,70)
                        ax.set_ylim(0, np.max(scan_denoise_tcp[:,2])+np.max(scan_denoise_tcp[:,2])*0.1)
                        plt.draw()
                        plt.pause(0.000000001)
                        # if scan_delta_h>5 and data_i>100:
                        #     print("scan delta h:",scan_delta_h)
                        #     input("Scan height too high, press Enter to continue...")
                        # get denoise scan
                        scan_exe_noise_remove.append(scan_denoise)
                        scan_exe_noise_remove_tcp.append(scan_denoise_tcp)
                        # get lambda and record height
                        scan_point_curve_dist = np.linalg.norm(curve[:,:2]-scan_point_location[:2],axis=1)
                        curve_index = np.argsort(scan_point_curve_dist)[0]
                        # if scan_point_curve_dist[curve_index]<1:
                        lam_scan = lam_relative[curve_index]
                        lam_scan_i = np.where(lam_split<=lam_scan)[0][-1]
                        if scan_delta_h<10:
                            lam_state_height[lam_scan_i].append(scan_delta_h)
                        curve_shift = scan_point_location[:2]-curve[curve_index][:2]
                        lam_curve_shift = np.vstack((lam_curve_shift,np.hstack((lam_scan,curve_shift))))
                ########################################

                #### for visualization
                plt.ioff()  # Turn off interactive mode
                plt.show()  # Keep the final frame displayed
                ###################

                for j, lam_height in enumerate(lam_state_height):
                    if len(lam_height) != 0:
                        print(f"Layer {i} Section {j} Height: {np.mean(lam_height)}")

                lam_curve_shift = lam_curve_shift[1:]
                lam_curve_shift = lam_curve_shift[lam_curve_shift[:,0]!=0]
                print(lam_curve_shift[0])

                print("Current mean shift x,y:",np.mean(lam_curve_shift[:,1]),np.mean(lam_curve_shift[:,2]))
                print("Current std shift x,y:",np.std(lam_curve_shift[:,1]),np.std(lam_curve_shift[:,2]))
                lam_curve_shift = lam_curve_shift[np.argsort(lam_curve_shift[:,0])]
                # 1d smoother
                lam_curve_shift_noise = np.where(np.abs(lam_curve_shift[:,2]-np.mean(lam_curve_shift[:,2]))>3*np.std(lam_curve_shift[:,2]))[0]
                print("Noise index:",lam_curve_shift_noise)
                lam_curve_shift[lam_curve_shift_noise,2] = np.mean(lam_curve_shift[:,2])
                lam_curve_shift_smooth_x = moving_average(lam_curve_shift[:,1],n=shift_pos_smoother,padding=True)
                lam_curve_shift_smooth_y = moving_average(lam_curve_shift[:,2],n=shift_pos_smoother,padding=True)
                plt.plot(lam_curve_shift[:,0])
                plt.title("Lambda")
                plt.show()
                plt.plot(lam_curve_shift[:,1])
                plt.plot(lam_curve_shift_smooth_x)
                plt.show()
                plt.plot(lam_curve_shift[:,2])
                plt.plot(lam_curve_shift_smooth_y)
                plt.show()
                # lam_curve_shift[:,1] = lam_curve_shift_smooth_x
                # lam_curve_shift[:,2] = lam_curve_shift_smooth_y

                ### for visualization
                fig, ax = plt.subplots()
                ax.set_xlim(-40, 40)  # Set x-axis limits
                ax.set_ylim(30, 120)  # Set y-axis limits
                # Initialize an empty line object
                line, = ax.plot([], [], '.')  # No data at start
                line_large, = ax.plot([], [], '.', ms=10)  # No data at start
                plt.ion()  # Turn on interactive mode
                plt.show()
                ######

                ####### welding motion ##########################
                lam_cur=0
                lam_split_i = 0
                v_cmd = dh2v_loglog(np.mean(lam_state_height[lam_split_i]),mode=layer_feedrate)
                v_cmd = np.clip(v_cmd,v_minimum,v_maximum)
                print("Initial Velocity:",round(v_cmd,2), "Mean dh:",np.mean(lam_state_height[lam_split_i]))
                for data_i in range(weld_start_index,len(weld_js_exe)):
                    loop_start=time.perf_counter()

                    ### get the next q commands
                    lam_cur+=v_cmd/streaming_rate # get the current lambda (path location)
                    # lam_idx=np.where(lam_relative>=lam_cur)[0][0] #get closest two indices and interpolate the joint angle
                    # ratio=(lam_cur-lam_relative[lam_idx-1])/(lam_relative[lam_idx]-lam_relative[lam_idx-1])
                    # q1=curve_js[lam_idx-1]*(1-ratio)+curve_js[lam_idx]*ratio
                    # q2=curve_js_cam[lam_idx-1]*(1-ratio)+curve_js_cam[lam_idx]*ratio
                    # q_pos=curve_js_positioner[lam_idx-1]*(1-ratio)+curve_js_positioner[lam_idx]*ratio
                    # q_cmd=np.hstack((q1,q2,q_pos))

                    # get the robot to the shifted position
                    # shifted_i = np.where(lam_curve_shift[:,0]>=lam_cur)[0][0]-1
                    # ### moving average for shifting
                    # if shifted_i < (shift_pos_smoother-1)/2:
                    #     shifted_xy = np.mean(lam_curve_shift[:shift_pos_smoother+1,1:3],axis=0)
                    # elif shifted_i > len(lam_curve_shift)-(shift_pos_smoother-1)/2:
                    #     shifted_xy = np.mean(lam_curve_shift[-shift_pos_smoother:,1:3],axis=0)
                    # else:
                    #     shifted_xy = np.mean(lam_curve_shift[shifted_i-int((shift_pos_smoother-1)/2):shifted_i+int((shift_pos_smoother-1)/2)+1,1:3],axis=0)

                    ### if welding start or end
                    pass
                    
                    ### update speed 
                    if weld_parts != 'base' and layer_count>=correction_layer: 
                        if lam_split_i < len(lam_split)-1 and lam_cur > lam_split[lam_split_i+1]:
                            lam_split_i += 1
                            if len(lam_state_height[lam_split_i]) == 0:
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
                            v_cmd_new = dh2v_loglog(np.mean(lam_state_height[lam_split_i]),mode=layer_feedrate)
                            if not np.isnan(v_cmd_new):
                                v_cmd = v_cmd_new
                            v_cmd = np.clip(v_cmd,v_minimum,v_maximum)
                            print("Update Velocity:",round(v_cmd,2), "Mean dh:",np.mean(lam_state_height[lam_split_i]))

                    ###update welding param
                    pass

                    ### scan online processing
                    while scan_process.accessing_key:
                        time.sleep(0.0000000000001)
                    scan_process.accessing_key = True
                    scan_process.raw_scan_pipe.append(deepcopy(scan_exe[data_i]))
                    scan_process.robot_q_pipe.append(deepcopy(weld_js_exe[data_i][np.array([1,2,3,4,5,6,13,14])])) # log robot joints (robot 1 and positioner)
                    scan_process.accessing_key = False
                    while len(scan_process.denoise_scan_pipe)==0: # wait for processing in simulation
                        time.sleep(0.0000000000001)
                    while len(scan_process.denoise_scan_pipe)!=0:
                        while scan_process.accessing_key:
                            time.sleep(0.0000000000001)
                        scan_process.accessing_key = True
                        scan_denoise = scan_process.denoise_scan_pipe.pop(0)
                        scan_denoise_tcp = scan_process.denoise_scan_tcp_pipe.pop(0)
                        scan_point_location = scan_process.point_location_pipe.pop(0)
                        scan_delta_h = scan_process.delta_h_pipe.pop(0)
                        scan_process.accessing_key = False
                        line.set_data(scan_denoise_tcp[:,1], scan_denoise_tcp[:,2])  # Update both x and y
                        line_large.set_data([scan_point_location[1]], [scan_point_location[2]-offset_z])  # Update both x and y
                        ax.set_xlim(30,70)
                        ax.set_ylim(0, np.max(scan_denoise_tcp[:,2])+np.max(scan_denoise_tcp[:,2])*0.1)
                        plt.draw()
                        plt.pause(0.000000001)
                        # get denoise scan
                        scan_exe_noise_remove.append(scan_denoise)
                        scan_exe_noise_remove_tcp.append(scan_denoise_tcp)
                        if lam_cur<lam_relative[int(-dist_weld_scan_index + 1/path_dl)]:
                            # get lambda and record height
                            scan_point_curve_dist = np.linalg.norm(curve[:,:2]-scan_point_location[:2],axis=1)
                            curve_index = np.argsort(scan_point_curve_dist)[0]
                            # if scan_point_curve_dist[curve_index]<1:
                            lam_scan = lam_relative[curve_index]
                            lam_scan_i = np.where(lam_split<=lam_scan)[0][-1]
                            lam_state_height[lam_scan_i].append(np.min((scan_delta_h,10)))
                            curve_shift = scan_point_location[:2]-curve[curve_index][:2]
                            if np.abs(curve_shift[1]-np.mean(lam_curve_shift[:,2]))>2*np.std(lam_curve_shift[:,2]):
                                curve_shift[1] = np.mean(lam_curve_shift[:,2])
                            lam_curve_shift = np.vstack((lam_curve_shift,np.hstack((lam_scan,curve_shift))))

                ########################################

                # final scan processing
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
                    curve_index = np.argsort(np.linalg.norm(curve[:,:2]-scan_point_location[:2],axis=1))[0]
                    lam_scan = lam_relative[curve_index]
                    lam_scan_i = np.where(lam_split<=lam_scan)[0][-1]
                    lam_state_height[lam_scan_i].append(scan_delta_h)
                    curve_shift = scan_point_location[:2]-curve[curve_index][:2]
                    lam_curve_shift = np.vstack((lam_curve_shift,np.hstack((lam_scan,curve_shift))))
                # stop scan process
                scan_process.end_denoise_thread_flag = True
                scan_dh_thread.join()

                #### for visualization
                plt.ioff()  # Turn off interactive mode
                plt.show()  # Keep the final frame displayed
                ###################

                for j, lam_height in enumerate(lam_state_height):
                    if len(lam_height) != 0:
                        print(f"Layer {i} Section {j} Height: {np.mean(lam_height)}")
                
                weld_js_exe = np.array(weld_js_exe)
                stamps_exe = deepcopy(weld_js_exe[:,0])
                ################### get layer increments ############################
                if adaptive_layer_height:
                    # single scan noise remove
                    if not scan_online_process:
                        scan_exe_noise_remove = []
                        for scan in scan_exe:
                            scan_noise_remove = scan_process.scan2dDenoise(deepcopy(scan).T,crop_min=[-40,30],crop_max=[40,200])
                            scan_exe_noise_remove.append(scan_noise_remove)
                        # 3D scan registration
                        pcd = scan_process.pcd_register_mti(scan_exe_noise_remove,weld_js_exe[:,np.append(np.arange(1,7),np.arange(13,15))],stamps_exe,flip=True,scanner='fuji')
                    else:
                        pcd = o3d.geometry.PointCloud()
                        for scan_tcp in scan_exe_noise_remove_tcp:
                            pcd_slice = o3d.geometry.PointCloud()
                            pcd_slice.points=o3d.utility.Vector3dVector(scan_tcp)
                            pcd_slice = pcd_slice.voxel_down_sample(voxel_size=0.05)
                            pcd += pcd_slice
                    visualize_pcd([pcd])
                    curve_planned_z = np.mean(curve[:,2])
                    curve_x_end = np.min(curve[:,0])
                    curve_x_start = np.max(curve[:,0])
                    curve_y = np.mean(curve[:,1])
                    z_height_start=curve_planned_z-3
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

                    print("Profile Height",profile_height[:,1])
                    mean_layer_height = np.nanmean(profile_height[:,1])
                    print("Mean Layer Height:",mean_layer_height)
                    # if weld_parts == 'base':
                    #     print("Expected next layer:", i+base_nom_incre) # baselayer uses base_nom_incre
                    # else:
                    #     # layer uses mean_layer_height/layer_resolution, 
                    #     # and add layer_nom_incre because scanner leads the welding
                    #     print("Expected next layer:", round((mean_layer_height-2*baselayer_resolution)/layer_resolution)+layer_nom_incre)
                ##########################################

                ### layer parameters update 
                layer_count += 1
                forward = not forward
            except:
                traceback.print_exc()
                break

if __name__ == '__main__':
    main()
    