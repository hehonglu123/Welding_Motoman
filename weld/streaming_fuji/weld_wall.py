import time, os, copy, sys, yaml, pathlib
import traceback
from copy import deepcopy
import numpy as np
import datetime
from motoman_def import *
from lambda_calc import *
from RobotRaconteur.Client import *
from weldRRSensor import *
from StreamingSend import *
from robotics_utils import *
sys.path.append('../')
from weld_dh2v import *

def main():
    
    weld_arcon = False
    fuji_scanon = False
    input_from_user = True

    ############## Robot definition ##############
    config_dir='../../config/'
    robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15)
    robot_scan=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'fujicam.csv',\
    pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
    positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_extended_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
		pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose.csv')

    positioner_joints = np.radians([-15,180])
    
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
    
    ######################################### RR Fujicam ########################################################
    if fuji_scanon:
        fujicam_url = 'rr+tcp://localhost:12181/?service=fujicam'
        def connect_failed(s, client_id, url, err):
            print ("Client connect failed: " + str(client_id.NodeID) + " url: " + str(url) + " error: " + str(err))
        sub=RRN.SubscribeService(fujicam_url)
        obj = sub.GetDefaultClientWait(2)		#connect, timeout=2s
        fuji_scan_wire=sub.SubscribeWire("lineProfile")
        sub.ClientConnectFailed += connect_failed
    
    ################## Read geometry data ##################
    data_dir = '../../data/wall_weld_test/'
    with open(data_dir+'sliced_meta.yml', 'r') as f:
        meta_data = yaml.safe_load(f)
    
    base_layer_num= meta_data['baselayernum']
    baselayer_resolution= meta_data['baselayer_resolution']
    layer_num = meta_data['layer_num']
    layer_resolution = meta_data['layer_resolution']

    job_offset=200
    # baselayer welding parameters
    base_feedrate = 250
    base_nom_incre = 1
    base_vel = 5
    # layer welding parameters
    layer_feedrate = 100
    layer_nom_height = 2
    layer_vel = 5
    layer_nom_incre = int(layer_nom_height/layer_resolution)

    feedrate_update_rate=1.	#Hz

    # start-end layers
    baselayer_start = 0
    baselayer_end = base_layer_num
    layer_start = 0
    layer_end = layer_num
    
    ################## Log data dir ##################
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-7]
    logdata_dir='../../data/wall_weld_test/weld_fujiscan_'+formatted_time+'/'

    weld_meta_data = {'well_arcon':weld_arcon, 'fuji_scanon':fuji_scanon, 'data_dir':data_dir, 'logdata_dir':logdata_dir\
        ,'base_layer_num':base_layer_num, 'baselayer_resolution':baselayer_resolution, 'layer_num':layer_num, 'layer_resolution':layer_resolution\
        ,'base_feedrate':base_feedrate, 'base_nom_incre':base_nom_incre, 'base_vel':base_vel\
        , 'layer_feedrate':layer_feedrate, 'layer_nom_incre':layer_nom_incre, 'layer_vel':layer_vel}

    # get robot 2 resting pose
    q_cur = deepcopy(SS.q_cur)
    r2_q_rest = q_cur[6:12]
    print("Robot 2 resting pose: ", np.degrees(r2_q_rest))

    ################## print layers ##################
    arc_off=True
    forward = True
    # for weld_parts in ['base','layer']:
    for weld_parts in ['base']:
        if weld_parts == 'base':
            weld_start = baselayer_start
            weld_end = baselayer_end
            nom_incre = base_nom_incre
            v_cmd = base_vel
            this_layer_feedrate = base_feedrate
        else:
            weld_start = layer_start
            weld_end = layer_end
            nom_incre = layer_nom_incre
            v_cmd = layer_vel
            this_layer_feedrate = layer_feedrate
        i = weld_start
        while i < weld_end:
            print(f'Welding {weld_parts} layer {i}')
            try:
                if forward:
                    curve_direction = 'forward'
                else:
                    curve_direction = 'backward'
                # read curve joint space data
                if weld_parts == 'base':
                    curve = np.loadtxt(data_dir+f'curve_sliced_relative/baselayer{i}_0.csv',delimiter=',')
                    curve_js = np.loadtxt(data_dir+f'curve_sliced_js/MA2010_base_js{i}_0.csv', delimiter=',')
                    curve_js_cam = np.loadtxt(data_dir+f'curve_sliced_js/MA1440_base_js{i}_{curve_direction}.csv', delimiter=',')
                    curve_js_positioner = np.loadtxt(data_dir+f'curve_sliced_js/D500B_base_js{i}_{curve_direction}.csv', delimiter=',')
                else:
                    curve = np.loadtxt(data_dir+f'curve_sliced_relative/slice{i}_0.csv',delimiter=',')
                    curve_js = np.loadtxt(data_dir+f'curve_sliced_js/MA2010_js{i}_0.csv', delimiter=',')
                    curve_js_cam = np.loadtxt(data_dir+f'curve_sliced_js/MA1440_js{i}_{curve_direction}.csv', delimiter=',')
                    curve_js_positioner = np.loadtxt(data_dir+f'curve_sliced_js/D500B_js{i}_{curve_direction}.csv', delimiter=',')

                # weld_start_js = curve_js[0]
                # weld_end_js = curve_js[-1]
                # min_index = np.argmin(np.linalg.norm(curve_js - curve_js_scan[0], axis=1))
                # if min_index == 0:
                #     curve_js = np.vstack((curve_js_scan[::-1],curve_js))
                #     curve = np.vstack((curve_scan[::-1],curve))
                # elif min_index == len(curve_js)-1:
                #     curve_js = np.vstack((curve_js,curve_js_scan))
                #     curve = np.vstack((curve,curve_scan))
                # else:
                #     assert False, 'No match found'
                
                if not forward:
                    curve = curve[::-1]
                lam_relative = calc_lam_cs(curve[:,:3])
                weld_start_idx = np.where(curve_js==weld_start_js)[0][0]
                weld_end_idx = np.where(curve_js==weld_end_js)[0][0]
                # print(weld_start_idx,weld_end_idx)
                
                print(f'Welding {weld_parts} layer {i}')
                if input_from_user:
                    input("Press Enter to continue...")
                else:
                    time.sleep(1)

                # move to start point with z +50
                z_offset = 35 # mm
                for z in np.arange(z_offset,0,-5): # a linear movement
                    T_start = robot.fwd(curve_js[0])
                    T_start.p[2] += z
                    curve_js_start_offset = robot.inv(T_start.p, T_start.R, last_joints=curve_js[0])[0]
                    q_start_offset = np.hstack((curve_js_start_offset, r2_q_rest, positioner_joints))
                    SS.jog2q(q_start_offset)
                # move to start point
                q_start = np.hstack((curve_js[0], r2_q_rest, positioner_joints))
                SS.jog2q(q_start)
                time.sleep(0.3)

                # start joints recording
                SS.start_recording()
                ####### welding motion ##########################
                lam_cur=0
                last_update_time=time.perf_counter()+5.
                q_cmd_all = []
                welding_cmd_all = []
                weld_js_exe = []
                scan_exe = []
                stamps_exe = []
                while lam_cur<lam_relative[-1] - v_cmd/SS.streaming_rate:
                    loop_start=time.perf_counter()

                    ### get the next q commands
                    lam_cur+=v_cmd/SS.streaming_rate # get the current lambda (path location)
                    lam_idx=np.where(lam_relative>=lam_cur)[0][0] #get closest two indices and interpolate the joint angle
                    ratio=(lam_cur-lam_relative[lam_idx-1])/(lam_relative[lam_idx]-lam_relative[lam_idx-1])
                    q1=curve_js[lam_idx-1]*(1-ratio)+curve_js[lam_idx]*ratio
                    q_cmd=np.hstack((q1,r2_q_rest,positioner_joints))

                    ### if welding start or end
                    if lam_cur >= lam_relative[weld_start_idx] and lam_cur<lam_relative[weld_end_idx] and arc_off:
                        print("stop for welding start")
                        time.sleep(1)
                        if weld_arcon:
                            print("Welding Start")
                            fronius_client.job_number = int(this_layer_feedrate/10+job_offset)
                            fronius_client.start_weld()
                        arc_off=False
                        print("continue welding")
                    if lam_cur >= lam_relative[weld_end_idx] and not arc_off:
                        print("stop for welding end")
                        if weld_arcon:
                            print("Welding End")
                            fronius_client.stop_weld()
                        arc_off=True
                        time.sleep(1)

                    ###update welding param
                    if time.perf_counter()-last_update_time>1./feedrate_update_rate:
                        welding_cmd_all.append(np.hstack((time.perf_counter(),i,v_cmd,this_layer_feedrate)))
                        ## TODO: update welding params
                        v_cmd = v_cmd
                        last_update_time=time.perf_counter()
                    
                    ### log data
                    weld_js_exe.append(deepcopy(SS.q_cur)) # log robot joints
                    if fuji_scanon:
                        wire_packet=fuji_scan_wire.TryGetInValue() # log fuji cam scanner data
                        valid_indices=np.where(wire_packet[1].I_data>1)[0]
                        valid_indices=np.intersect1d(valid_indices,np.where(np.abs(wire_packet[1].Z_data)>50)[0])
                        line_profile=np.hstack((wire_packet[1].Y_data[valid_indices].reshape(-1,1),wire_packet[1].Z_data[valid_indices].reshape(-1,1)))
                        scan_exe.append(line_profile)
                    stamps_exe.append(time.perf_counter()) # log time stamps

                    ### sent position Command to the robot
                    q_cmd_all.append(np.hstack((time.perf_counter(),i,q_cmd)))
                    if lam_cur>lam_relative[-1]-v_cmd/SS.streaming_rate:
                        SS.position_cmd(q_cmd)
                    else:
                        SS.position_cmd(q_cmd,loop_start)

                ### welding end
                if weld_arcon:
                    fronius_client.stop_weld()
                arc_off=True
                js_recording = SS.stop_recording()
                ########################################

                # move to end point with z +50
                time.sleep(0.3)
                z_offset = 35 # mm
                for z in np.arange(0,z_offset+1,5): # a linear movement
                    T_end = robot.fwd(curve_js[-1])
                    T_end.p[2] += z
                    curve_js_end_offset = robot.inv(T_end.p, T_end.R, last_joints=curve_js[-1])[0]
                    q_end_offset = np.hstack((curve_js_end_offset, r2_q_rest, positioner_joints))
                    SS.jog2q(q_end_offset)

                ### save data
                if not os.path.exists(logdata_dir):
                    os.makedirs(logdata_dir)
                # save meta data
                with open(logdata_dir+'weld_meta_data.yml', 'w') as f:
                    yaml.dump(weld_meta_data, f)
                if weld_parts == 'base':
                    layer_name = 'baselayer'+str(i)
                    pathlib.Path(logdata_dir+layer_name).mkdir(parents=True, exist_ok=True)
                    np.savetxt(logdata_dir+layer_name+f'/baselayer{i}_timestamps_exe.csv', stamps_exe, delimiter=',')
                    np.savetxt(logdata_dir+layer_name+f'/baselayer{i}_weld_js_exe.csv', weld_js_exe, delimiter=',')
                    np.savetxt(logdata_dir+layer_name+f'/baselayer{i}_weld_cmd.csv', welding_cmd_all, delimiter=',')
                    np.savetxt(logdata_dir+layer_name+f'/baselayer{i}_weld_js_recording.csv', js_recording, delimiter=',')
                    if fuji_scanon:
                        with open(logdata_dir+layer_name+f'/baselayer{i}_scan_exe.pickle', 'wb') as file:
                            pickle.dump(scan_exe, file)
                else:
                    layer_name = 'layer'+str(i)
                    pathlib.Path(logdata_dir+layer_name).mkdir(parents=True, exist_ok=True)
                    np.savetxt(logdata_dir+layer_name+f'/layer{i}_timestamps_exe.csv', stamps_exe, delimiter=',')
                    np.savetxt(logdata_dir+layer_name+f'/layer{i}_weld_js_exe.csv', weld_js_exe, delimiter=',')
                    np.savetxt(logdata_dir+layer_name+f'/layer{i}_weld_cmd.csv', welding_cmd_all, delimiter=',')
                    np.savetxt(logdata_dir+layer_name+f'/layer{i}_weld_js_recording.csv', js_recording, delimiter=',')
                    if fuji_scanon:
                        with open(logdata_dir+layer_name+f'/layer{i}_scan_exe.pickle', 'wb') as file:
                            pickle.dump(scan_exe, file)

                ### layer parameters update 
                forward = not forward
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