from copy import deepcopy
from pathlib import Path
import pickle
import sys
sys.path.append('../toolbox/')
sys.path.append('../scan/scan_tools/')
sys.path.append('../scan/scan_plan/')
sys.path.append('../scan/scan_process/')
from robot_def import *
from scan_utils import *
from scan_continuous import *
from scanPathGen import *
from scanProcess import *
from weldCorrectionStrategy import *
from WeldSend import *
from dx200_motion_program_exec_client import *

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import glob
import yaml
from math import ceil,floor

weld_timestamp=[]
weld_voltage=[]
weld_current=[]
weld_feedrate=[]
weld_energy=[]

def clean_weld_record():
    global weld_timestamp, weld_voltage, weld_current, weld_feedrate, weld_energy
    weld_timestamp=[]
    weld_voltage=[]
    weld_current=[]
    weld_feedrate=[]
    weld_energy=[]

def wire_cb(sub, value, ts):
    global weld_timestamp, weld_voltage, weld_current, weld_feedrate, weld_energy

    weld_timestamp.append(value.ts['microseconds'][0])
    weld_voltage.append(value.welding_voltage)
    weld_current.append(value.welding_current)
    weld_feedrate.append(value.wire_speed)
    weld_energy.append(value.welding_energy)

zero_config=np.zeros(6)
# 0. robots.
config_dir='../config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'torch.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA1440_marker_config.yaml')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config.yaml')

Table_home_T = positioner.fwd(np.radians([-15,180]))
T_S1TCP_R1Base = np.linalg.inv(np.matmul(positioner.base_H,H_from_RT(Table_home_T.R,Table_home_T.p)))
T_R1Base_S1TCP = np.linalg.inv(T_S1TCP_R1Base)

#### change base H to calibrated ones ####
robot_scan.base_H = H_from_RT(robot_scan.T_base_basemarker.R,robot_scan.T_base_basemarker.p)
positioner.base_H = H_from_RT(positioner.T_base_basemarker.R,positioner.T_base_basemarker.p)
T_to_base = Transform(np.eye(3),[0,0,-380])
positioner.base_H = np.matmul(positioner.base_H,H_from_RT(T_to_base.R,T_to_base.p))

# exit()

#### data directory
dataset='cup/'
sliced_alg='circular_slice_shifted/'
curve_data_dir = '../data/'+dataset+sliced_alg

current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-7]
data_dir=curve_data_dir+'../weld_scan_'+formatted_time+'/'

#### welding spec, goal
with open(curve_data_dir+'slicing.yml', 'r') as file:
    slicing_meta = yaml.safe_load(file)
line_resolution = slicing_meta['line_resolution']

weld_mode=160
des_dh = 1.4
des_v = round(dh2v_loglog(des_dh,weld_mode),1)
print("The Desired speed (according to desired h",des_dh,"will be",\
      des_v,"mm/sec")
des_dw = 4
waypoint_distance=1.625 	###waypoint separation (calculate from 40moveL/95mm, where we did the test)
layer_height_num=int(des_dh/line_resolution) # preplanned
layer_width_num=int(des_dw/line_resolution) # preplanned
des_job=200

### preplanned v,height for first few layer
planned_v=[5,5]
planned_layer=[0,layer_height_num]

# 2. Scanning parameters
### scan parameters
scan_speed=10 # scanning speed (mm/sec)
scan_stand_off_d = 95 ## mm
Rz_angle = np.radians(0) # point direction w.r.t welds
Ry_angle = np.radians(0) # rotate in y a bit
bounds_theta = np.radians(1) ## circular motion at start and end
all_scan_angle = np.radians([0]) ## scan angle
q_init_table=np.radians([-15,200]) ## init table
mti_Rpath = np.array([[ -1.,0.,0.],   
                        [ 0.,1.,0.],
                        [0.,0.,-1.]])

# ## rr drivers and all other drivers
# robot_client=MotionProgramExecClient()
ws=WeldSend(robot_client)
# # weld state logging
# sub = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
# obj = sub.GetDefaultClientWait(3)  # connect, timeout=30s
# welder_state_sub = sub.SubscribeWire("welder_state")
# welder_state_sub.WireValueChanged += wire_cb
# # MTI connect to RR
# mti_client = RRN.ConnectService("rr+tcp://192.168.55.10:60830/?service=MTI2D")
# mti_client.setExposureTime("25")
###################################
start_feedback=2
save_weld_record=True
save_output_points=True

all_profile_height=None
curve_sliced_relative=None
all_last_curve_relative=None
layer=0
last_layer=-1
layer_count=0
mean_h=0
mean_layer_dh=None
while True:
    print("Layer Count:",layer_count)
    ####### Decide which layer to print #######
    if layer_count!=0 and layer_count<start_feedback:
        layer+=layer_height_num
    else:
        dlayer = int(round(mean_layer_dh/line_resolution)) # find the "delta layer" using dh
        last_layer = layer # update last layer
        layer = layer+dlayer # update layer

    print("Print Layer:",layer)
    ####################DETERMINE CURVE ORDER##############################################
    all_curve_relative=[]
    num_sections=len(glob.glob(curve_data_dir+'curve_sliced_relative/slice'+str(layer)+'_*.csv'))
    for x in range(0,num_sections,layer_width_num):
        #### welding
        if layer>=0 and True:

            # 1. The curve path in "positioner tcp frame"
            # Load nominal path given the layer
            curve_sliced_js=np.loadtxt(curve_data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
            if len(curve_sliced_js)<2:
                continue
            positioner_js=np.loadtxt(curve_data_dir+'curve_sliced_js/D500B_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
            curve_sliced_relative=np.loadtxt(curve_data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')

            #### convert to R1 and S1 motion
            lam1=calc_lam_js(curve_sliced_js,robot_weld)
            lam2=calc_lam_js(positioner_js,positioner)
            lam_relative=calc_lam_cs(curve_sliced_relative)

            num_points_layer=max(2,int(lam_relative[-1]/waypoint_distance))

            ###find which end to start depending on how close to joint limit
            if positioner.upper_limit[1]-q_prev[1]>q_prev[1]-positioner.lower_limit[1]:
                breakpoints=np.linspace(0,len(curve_sliced_js)-1,num=num_points_layer).astype(int)
            else:
                breakpoints=np.linspace(len(curve_sliced_js)-1,0,num=num_points_layer).astype(int)

            #### Correction ####
            if (layer_count<start_feedback): # no correction
                this_weld_v=planned_v[layer_count]
            else: # start correction after "start_feedback"
                if all_profile_height is None:
                    last_num_sections=len(glob.glob(curve_data_dir+'curve_sliced_relative/slice'+str(last_layer)+'_*.csv'))
                    all_profile_height=[]
                    all_last_curve_relative=[]
                    for x in range(0,last_num_sections,layer_width_num):
                        all_last_curve_relative.extend(np.loadtxt(curve_data_dir+'curve_sliced_relative/slice'+str(last_layer)+'_'+str(x)+'.csv',delimiter=','))
                        layer_data_dir=data_dir+'layer_'+str(layer)+'_'+str(x)+'/'
                        out_scan_dir = layer_data_dir+'scans/'
                        all_profile_height.extend(np.load(out_scan_dir+'height_profile.npy'))

                ## parameters
                #### correction strategy
                this_weld_v,all_dh=\
                    strategy_4(all_profile_height,des_dh,curve_sliced_relative,all_last_curve_relative,breakpoints)
                
            ####################
            print("dh:",all_dh)
            print("Nominal V:",des_v)
            print("Corrected V:",this_weld_v)
            print(len(curve_sliced_relative))

            ## use vel=1 and times the desired speed
            s1_all,s2_all=calc_individual_speed(1,lam1,lam2,lam_relative,breakpoints)
            s1_all=np.multiply(s1_all,this_weld_v)
            s2_all=np.multiply(s2_all,this_weld_v)

            ###move to intermidieate waypoint for collision avoidance if multiple section
            # if num_sections!=num_sections_prev:
            waypoint_pose=robot_weld.fwd(curve_sliced_js[breakpoints[0]])
            waypoint_pose.p[-1]+=50
            q1=robot_weld.inv(waypoint_pose.p,waypoint_pose.R,curve_sliced_js[breakpoints[0]])[0]
            q2=positioner_js[breakpoints[0]]
            ws.jog_dual(robot_weld,positioner,q1,q2)

            ######################################################
            ########### Do welding #############
            q1_all=[curve_sliced_js[breakpoints[0]]]
            q2_all=[positioner_js[breakpoints[0]]]
            v1_all=[1]
            v2_all=[10]
            primitives=['movej']
            for j in range(1,len(breakpoints)):
                q1_all.append(curve_sliced_js[breakpoints[j]])
                q2_all.append(positioner_js[breakpoints[j]])
                v1_all.append(max(s1_all[j-1],0.1))
                positioner_w=vd_relative/np.linalg.norm(curve_sliced_relative[breakpoints[j]][:2])
                v2_all.append(min(100,100*positioner_w/positioner.joint_vel_limit[1]))
                primitives.append('movel')

            q_prev=positioner_js[breakpoints[-1]]

            ####DATA LOGGING
            if save_weld_record:
                clean_weld_record()
            rob_stamps,rob_js_exe,_,_=ws.weld_segment_dual(primitives,robot_weld,positioner,q1_all,q2_all,v1_all,v2_all,cond_all=[200],arc=True)
            if save_weld_record:
                Path(data_dir).mkdir(exist_ok=True)
                layer_data_dir=data_dir+'layer_'+str(layer)+'_'+str(x)+'/'
                Path(layer_data_dir).mkdir(exist_ok=True)
                np.savetxt(layer_data_dir + 'welding.csv',
                            np.array([weld_timestamp, weld_voltage, weld_current, weld_feedrate, weld_energy]).T, delimiter=',',
                            header='timestamp,voltage,current,feedrate,energy', comments='')
                np.savetxt(layer_data_dir + 'weld_js_exe.csv',rob_js_exe,delimiter=',')
                np.savetxt(layer_data_dir + 'weld_robot_stamps.csv',rob_stamps,delimiter=',')
            
            all_curve_relative.append(curve_sliced_relative)

    #### scanning
    if True:
        if len(all_curve_relative) ==0:
            num_sections=len(glob.glob(curve_data_dir+'curve_sliced_relative/slice'+str(layer)+'_*.csv'))
            all_curve_relative=[]
            for x in range(0,num_sections,layer_width_num):
                all_curve_relative.append(np.loadtxt(curve_data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=','))

        all_profile_height=[]
        all_last_curve_relative=[]
        section_count=0
        for x in range(0,num_sections,layer_width_num):
            curve_sliced_relative=np.loadtxt(curve_data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
            positioner_js=np.loadtxt(curve_data_dir+'curve_sliced_js/D500B_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
            ## reverse the welding path
            curve_sliced_relative=curve_sliced_relative[::-1]
            positioner_js=positioner_js[::-1]
        
        # for curve_sliced_relative in all_curve_relative:    
        
            ### scanning path module
            spg = ScanPathGen(robot_scan,positioner,scan_stand_off_d,Rz_angle,Ry_angle,bounds_theta)
            
            try:
                q_out1=np.loadtxt(curve_data_dir+'curve_scan_js/MA1440_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
                q_out2=np.loadtxt(curve_data_dir+'curve_scan_js/D500B_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
                scan_p=np.loadtxt(curve_data_dir+'curve_scan_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
            except:
                # generate scan path
                scan_p,scan_R,q_out1,q_out2=spg.gen_scan_path([curve_sliced_relative],[0],all_scan_angle,\
                                    solve_js_method=1,q_init_table=positioner_js[0],R_path=mti_Rpath,scan_path_dir=None)
            # generate motion program
            q_bp1,q_bp2,s1_all,s2_all=spg.gen_motion_program(q_out1,q_out2,scan_p,scan_speed,init_sync_move=0)
            #######################################

            ######## scanning motion #########
            ### execute motion ###
            # input("Press Enter and move to scanning startint point")

            ## move to start
            to_start_speed=7
            mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot_scan.pulse2deg,pulse2deg_2=positioner.pulse2deg)
            target2=['MOVJ',np.degrees(q_bp2[0][0]),to_start_speed]
            mp.MoveJ(np.degrees(q_bp1[0][0]), to_start_speed, 0, target2=target2)
            robot_client.execute_motion_program(mp)

            input("Press Enter to start moving and scanning")

            ## motion start
            mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot_scan.pulse2deg,pulse2deg_2=positioner.pulse2deg)
            # calibration motion
            target2=['MOVJ',np.degrees(q_bp2[1][0]),s2_all[0]]
            mp.MoveL(np.degrees(q_bp1[1][0]), scan_speed, 0, target2=target2)
            # routine motion
            for path_i in range(2,len(q_bp1)-1):
                target2=['MOVJ',np.degrees(q_bp2[path_i][0]),s2_all[path_i]]
                mp.MoveL(np.degrees(q_bp1[path_i][0]), s1_all[path_i], target2=target2)
            target2=['MOVJ',np.degrees(q_bp2[-1][0]),s2_all[-1]]
            mp.MoveL(np.degrees(q_bp1[-1][0]), s1_all[-1], 0, target2=target2)

            robot_client.execute_motion_program_nonblocking(mp)
            ###streaming
            robot_client.StartStreaming()
            start_time=time.time()
            state_flag=0
            joint_recording=[]
            robot_stamps=[]
            mti_recording=[]
            r_pulse2deg = np.append(robot_scan.pulse2deg,positioner.pulse2deg)
            while True:
                if state_flag & 0x08 == 0 and time.time()-start_time>1.:
                    break
                res, data = robot_client.receive_from_robot(0.01)
                if res:
                    joint_angle=np.radians(np.divide(np.array(data[26:34]),r_pulse2deg))
                    state_flag=data[16]
                    joint_recording.append(joint_angle)
                    timestamp=data[0]+data[1]*1e-9
                    robot_stamps.append(timestamp)
                    ###MTI scans YZ point from tool frame
                    mti_recording.append(deepcopy(np.array([mti_client.lineProfile.X_data,mti_client.lineProfile.Z_data])))
            robot_client.servoMH(False)
            
            mti_recording=np.array(mti_recording)
            q_out_exe=joint_recording

            # input("Press Enter to Move Home")
            # move robot to home
            q2=np.zeros(6)
            q2[0]=90
            q3=[-15,180]
            mp=MotionProgram(ROBOT_CHOICE='RB2',pulse2deg=robot_scan.pulse2deg)
            mp.MoveJ(q2,5,0)
            robot_client.execute_motion_program(mp)
            mp=MotionProgram(ROBOT_CHOICE='ST1',pulse2deg=positioner.pulse2deg)
            mp.MoveJ(q3,10,0)
            robot_client.execute_motion_program(mp)
            #####################
            # exit()

            print("Total exe len:",len(q_out_exe))
            if save_output_points:
                layer_data_dir=data_dir+'layer_'+str(layer)+'_'+str(section_count)+'/'
                out_scan_dir = layer_data_dir+'scans/'
                Path(out_scan_dir).mkdir(exist_ok=True)
                ## save traj
                # save poses
                np.savetxt(out_scan_dir + 'scan_js_exe.csv',q_out_exe,delimiter=',')
                np.savetxt(out_scan_dir + 'scan_robot_stamps.csv',robot_stamps,delimiter=',')
                with open(out_scan_dir + 'mti_scans.pickle', 'wb') as file:
                    pickle.dump(mti_recording, file)
                print('Total scans:',len(mti_recording))
            ########################

            #### scanning process: processing point cloud and get h
            crop_extend=10
            crop_min=tuple(np.min(curve_sliced_relative[:][:3],axis=0)-crop_extend)
            crop_max=tuple(np.max(curve_sliced_relative[:][:3],axis=0)+crop_extend)
            scan_process = ScanProcess(robot_scan,positioner)
            pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps)
            pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
                                                min_bound=crop_min,max_bound=crop_max,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=100)
            profile_height = scan_process.pcd2dh(pcd,curve_sliced_relative)

            all_profile_height.extend(profile_height)
            all_last_curve_relative.extend(curve_sliced_relative)
            
            if save_output_points:
                o3d.io.write_point_cloud(out_scan_dir+'processed_pcd.pcd',pcd)
                np.save(out_scan_dir+'height_profile.npy',profile_height)
            # visualize_pcd([pcd])
            # plt.scatter(profile_height[:,0],profile_height[:,1])
            # plt.show()
            # exit()

            section_count+=layer_width_num

    ## increase layer count
    layer_count+=1

print("Welding End!!")