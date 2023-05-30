from copy import deepcopy
from pathlib import Path
import pickle
import sys
sys.path.append('../toolbox/')
sys.path.append('scan_tools/')
sys.path.append('scan_plan/')
sys.path.append('scan_process/')
from robot_def import *
from scan_utils import *
from scan_continuous import *
from scanPathGen import *
from scanProcess import *
from dx200_motion_program_exec_client import *

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np

weld_timestamp=[]
weld_weld_voltage=[]
weld_weld_current=[]
weld_weld_feedrate=[]
weld_weld_energy=[]

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
    weld_voltage.append(value.welding_weld_voltage)
    weld_current.append(value.welding_weld_current)
    weld_feedrate.append(value.wire_speed)
    weld_energy.append(value.welding_weld_energy)

def robot_weld_path_gen(all_layer_z,forward_flag,base_layer):
    R=np.array([[ 0.7071, -0.7071, -0.    ],
			[-0.7071, -0.7071,  0.    ],
			[-0.,      0.,     -1.    ]])
    x0 =  1684	# Origin x coordinate
    y0 = -1179 + 428	# Origin y coordinate
    z0 = -245   # 10 mm distance to base

    weld_p=[]
    if base_layer: # base layer
        weld_p.append([x0 - 33, y0 - 20, z0+10])
        weld_p.append([x0 - 33, y0 - 20, z0])
        weld_p.append([x0 - 33, y0 - 105 , z0])
        weld_p.append([x0 - 33, y0 - 105 , z0+10])
    else: # top layer
        weld_p.append([x0 - 33, y0 - 30, z0+10])
        weld_p.append([x0 - 33, y0 - 30, z0])
        weld_p.append([x0 - 33, y0 - 95 , z0])
        weld_p.append([x0 - 33, y0 - 95 , z0+10])

    if not forward_flag:
        weld_p = weld_p[::-1]

    all_path_T=[]
    for layer_z in all_layer_z:
        path_T=[]
        for p in weld_p:
            path_T.append(Transform(R,p))

        all_path_T.append(path_T)
    
    return all_path_T

zero_config=np.zeros(6)
# 0. robots. Note use "(robot)_pose_mocapcalib.csv"
config_dir='../config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'torch.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'scanner_tcp2.csv',\
	base_transformation_file=config_dir+'MA1440_pose_mocapcalib.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA1440_marker_config.yaml',tool_marker_config_file=config_dir+'scanner_marker_config.yaml')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose_mocapcalib.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config.yaml')

Table_home_T = positioner.fwd(np.radians([-15,180]))
T_S1TCP_R1Base = np.linalg.inv(np.matmul(positioner.base_H,H_from_RT(Table_home_T.R,Table_home_T.p)))

forward_flag = True
base_layer = True

final_height=30
weld_z_height=[0,5,10] # two base layer height to first top layer
weld_z_height=np.append(weld_z_height,np.arange(weld_z_height[-1],final_height+1,1))
job_number=[250,250]
job_number=np.append(job_number,np.ones(len(weld_z_height)-1)*160)
weld_velocity=[5,5]
weld_v=4
for i in range(len(weld_z_height)-2):
    weld_velocity.append(weld_v)
    if weld_v==weld_velocity[-2]:
        weld_v+=2
correction_thres=1.2 ## mm
save_weld_record=True

current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-7]
data_dir='../data/wall_weld_test/weld_scan_'+formatted_time+'/'

for i in range(len(weld_z_height)):

    if i>=2:
        base_layer=False
    this_z_height=weld_z_height[i]
    this_job_number=job_number[i]
    this_weld_v=weld_velocity[i]

    # 1. The curve path in "positioner tcp frame"
    ######### enter your wanted z height #######
    all_layer_z = [this_z_height]
    ###########################################
    all_path_T = robot_weld_path_gen(all_layer_z,forward_flag,base_layer) # this is your path
    path_T=all_path_T[0]
    curve_sliced_relative=[]
    for path_p in path_T:
        this_p = np.matmul(T_S1TCP_R1Base[:3,:3],path_p.p)+T_S1TCP_R1Base[:3,-1]
        this_n = np.matmul(T_S1TCP_R1Base[:3,:3],path_p.R[:,-1])
        curve_sliced_relative.append(np.append(this_p,this_n))
    curve_sliced_relative=curve_sliced_relative[1:-1] # the start and end is for collision prevention

    # 2. Scanning parameters
    ### scan parameters
    scan_speed=10 # scanning speed (mm/sec)
    scan_stand_off_d = 70 ## mm
    Rz_angle = np.radians(0) # point direction w.r.t welds
    Ry_angle = np.radians(0) # rotate in y a bit
    bounds_theta = np.radians(10) ## circular motion at start and end
    all_scan_angle = np.radians([0]) ## scan angle
    q_init_table=np.radians([-30,0]) ## init table
    save_output_points = True

    ### scanning path module
    spg = ScanPathGen(robot_scan,positioner,scan_stand_off_d,Rz_angle,Ry_angle,bounds_theta)
    # generate scan path
    scan_p,scan_R,q_out1,q_out2=spg.gen_scan_path([curve_sliced_relative],[0],all_scan_angle,\
                    solve_js_method=0,q_init_table=q_init_table,scan_path_dir=None)
    # generate motion program
    q_bp1,q_bp2,s1_all,s2_all=spg.gen_motion_program(q_out1,q_out2,scan_p,scan_speed,init_sync_move=0)

    ######################################################
    ########### Do welding #############
    #### Correction ####
    # TODO: Add fitering if near threshold
    if i>1:
        h_largest = np.max(profile_height[:,1])
        if h_largest-profile_height[0,1]>correction_thres:
            correct_part=True
        else:
            correct_part=False
        correction_T=[profile_height[0]]
        for sample_i in range(len(profile_height)):
            if correct_part:
                if h_largest-profile_height[sample_i][1]<correction_thres:
                    correction_T.append(profile_height[sample_i])
            else:
                if h_largest-profile_height[sample_i][1]>correction_thres:
                    correction_T.append(profile_height[sample_i])
    ####################
    path_q = []
    for tcp_T in path_T:
        path_q.append(robot_weld.inv(tcp_T.p,tcp_T.R,zero_config)[0])
    
    input("Press Enter and move to weld starting point.")
    mp = MotionProgram(ROBOT_CHOICE='RB1', pulse2deg=robot_weld.pulse2deg)
    mp.MoveJ(np.degrees(path_q[0]), 3, 0)
    mp.MoveL(np.degrees(path_q[1]), 10, 0)

    # weld state logging
    sub = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
    obj = sub.GetDefaultClientWait(3)  # connect, timeout=30s
    welder_state_sub = sub.SubscribeWire("welder_state")
    welder_state_sub.WireValueChanged += wire_cb
    mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=robot_weld.pulse2deg)
    mp.setArc(select, this_job_number)
    mp.MoveL(np.degrees(q3), this_weld_v, 0)
    mp.setArc(False)
    clean_weld_record()
    rob_stamps,rob_js_exe,_,_= robot_client.execute_motion_program(mp)
    if save_weld_record:
        layer_data_dir=data_dir+'layer_'+str(i)+'/'
        Path(layer_data_dir).mkdir(exist_ok=True)
        np.savetxt(layer_data_dir + 'welding.csv',
                    np.array([weld_timestamp, weld_voltage, weld_current, weld_feedrate, weld_energy]).T, delimiter=',',
                    header='timestamp,voltage,current,feedrate,energy', comments='')

    ######################################################

    ######## scanning motion #########
    ### execute motion ###
    robot_client=MotionProgramExecClient()
    input("Press Enter and move to scanning startint point")

    ## move to start
    to_start_speed=3
    mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot_scan.pulse2deg,pulse2deg_2=positioner.pulse2deg)
    target2=['MOVJ',np.degrees(q_bp2[0][0]),to_start_speed]
    mp.MoveJ(np.degrees(q_bp1[0][0]), to_start_speed, 0, target2=target2)
    robot_client.execute_motion_program(mp)

    input("Press Enter to start moving and scanning")

    ###MTI connect to RR
    mti_client = RRN.ConnectService("rr+tcp://192.168.55.10:60830/?service=MTI2D")
    mti_client.setExposureTime("25")

    ## motion start
    mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot_scan.pulse2deg,pulse2deg_2=turn_table.pulse2deg)
    # calibration motion
    target2=['MOVJ',np.degrees(q_bp2[1][0]),10]
    mp.MoveL(np.degrees(q_bp1[1][0]), scan_speed, 0, target2=target2)
    # routine motion
    for path_i in range(2,len(q_bp1)-1):
        target2=['MOVJ',np.degrees(q_bp2[path_i][0]),s2_all[path_i]]
        mp.MoveL(np.degrees(q_bp1[path_i][0]), s1_all[path_i], target2=target2)
    target2=['MOVJ',np.degrees(q_bp2[-1][0]),10]
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

    print(np.degrees(joint_recording[-10:]))

    input("Press Enter to Move Home")
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
        out_scan_dir = layer_data_dir+'scans/'
        ## save traj
        Path(out_scan_dir).mkdir(exist_ok=True)
        # save poses
        np.savetxt(out_scan_dir + 'scan_js_exe.csv',q_out_exe,delimiter=',')
        np.savetxt(out_scan_dir + 'robot_stamps.csv',robot_stamps,delimiter=',')
        with open(out_scan_dir + 'mti_scans.pickle', 'wb') as file:
            pickle.dump(mti_recording, file)
        print('Total scans:',len(mti_recording))

    #### scanning process: processing point cloud and get h
    scan_process = scanProcess(robot_scan,positioner,static_positioner_q=q_init_table)

    pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps)
    pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=10,std_ratio=0.85,cluster_based_outlier_remove=False)
    z_height_start=this_z_height
    profile_height = scan_process.pcd2height(pcd,z_height_start)

    forward_flag=not forward_flag