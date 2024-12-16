import time, os, copy, sys, yaml
import traceback
from copy import deepcopy
import numpy as np
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
        fronius_client = fronius_sub.GetDefaultClientWait(1)      #connect, timeout=30s
        hflags_const = RRN.GetConstants("experimental.fronius", fronius_client)["WelderStateHighFlags"]
        fronius_client.prepare_welder()
    
    ################## Read geometry data ##################
    data_dir = '../../data/wall_weld_test/'
    with open(data_dir+'sliced_meta.yml', 'r') as f:
        meta_data = yaml.safe_load(f)
    
    base_layer_num= meta_data['baselayernum']
    baselayer_resolution= meta_data['baselayer_resolution']
    layer_num = meta_data['layer_num']
    layer_resolution = meta_data['layer_resolution']

    # welding parameters
    base_feedrate = 250
    base_vel = 5
    layer_feedrate = 100
    layer_nom_height = 2.4
    layer_nom_incre = int(layer_nom_height/layer_resolution)
    job_offset=200

    feedrate_update_rate=1.	#Hz
    input_from_user = True

    # start-end layers
    baselayer_start = 0
    baselayer_end = base_layer_num
    layer_start = 0
    layer_end = layer_num

    # get robot 2 resting pose
    q_cur = deepcopy(SS.q_cur)
    r2_q_rest = q_cur[6:12]
    print("Robot 2 resting pose: ", np.degrees(r2_q_rest))

    ################## print base layer ##################
    arc_off=True
    forward = True
    for i in range(baselayer_start,baselayer_end):
        try:
            # read curve joint space data
            curve = np.loadtxt(data_dir+f'curve_sliced_relative/baselayer{i}_0.csv',delimiter=',')
            curve_scan = np.loadtxt(data_dir+f'curve_sliced_relative/baselayer{i}_scanOnly.csv',delimiter=',')
            curve_js = np.loadtxt(data_dir+f'curve_sliced_js/MA2010_base_js{i}_0.csv', delimiter=',')
            curve_js_scan = np.loadtxt(data_dir+f'curve_sliced_js/MA2010_base_js{i}_scanOnly.csv', delimiter=',')
            weld_start_js = curve_js[0]
            weld_end_js = curve_js[-1]
            min_index = np.argmin(np.linalg.norm(curve_js - curve_js_scan[0], axis=1))
            if min_index == 0:
                curve_js = np.vstack((curve_js_scan[::-1],curve_js))
                curve = np.vstack((curve_scan[::-1],curve))
            elif min_index == len(curve_js)-1:
                curve_js = np.vstack((curve_js,curve_js_scan))
                curve = np.vstack((curve,curve_scan))
            else:
                assert False, 'No match found'
            
            if not forward:
                curve_js = curve_js[::-1]
                curve = curve[::-1]
                weld_start_dummy = deepcopy(weld_start_js)
                weld_start_js = deepcopy(weld_end_js)
                weld_end_js = deepcopy(weld_start_dummy)
            lam_relative = calc_lam_cs(curve[:,:3])
            weld_start_idx = np.where(curve_js==weld_start_js)[0][0]
            weld_end_idx = np.where(curve_js==weld_end_js)[0][0]
            print(weld_start_idx,weld_end_idx)
            
            if input_from_user:
                print(f"Base layer {i} will be printed. Press enter to continue.")
                input()

            # move to start point
            q_start = np.hstack((curve_js[0], r2_q_rest, positioner_joints))
            SS.jog2q(q_start)

            # start joints recording
            SS.start_recording()
            ####### welding motion ##########################
            v_cmd = base_vel
            lam_cur=0
            last_update_time=time.perf_counter()+5.
            q_cmd_all = []
            welding_cmd_all = []
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
                        fronius_client.job_number = int(layer_feedrate/10+job_offset)
                        fronius_client.start_weld()
                    arc_off=False
                    print("continue welding")
                if lam_cur >= lam_relative[weld_end_idx] and not arc_off:
                    if weld_arcon:
                        print("Welding End")
                        fronius_client.stop_weld()
                    arc_off=True
                    time.sleep(1)

                ###update welding param
                if time.perf_counter()-last_update_time>1./feedrate_update_rate:
                    welding_cmd_all.append(np.hstack((time.perf_counter(),i,v_cmd,layer_feedrate)))
                    ## TODO: update welding params
                    v_cmd = v_cmd
                    last_update_time=time.perf_counter()

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