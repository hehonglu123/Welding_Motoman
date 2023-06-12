from copy import deepcopy
from pathlib import Path
import pickle
import sys
sys.path.append('../../toolbox/')
sys.path.append('../scan_tools/')
sys.path.append('../scan_plan/')
sys.path.append('../scan_process/')
from robot_def import *
from scan_utils import *
from scan_continuous import *
from scanPathGen import *
from scanProcess import *
from weldCorrectionStrategy import *
from dx200_motion_program_exec_client import *

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np

zero_config=np.zeros(6)
config_dir='../../config/'
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA1440_marker_config.yaml')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config.yaml')

data_dir='../../data/lego_brick/mti/'
Path(data_dir).mkdir(exist_ok=True)
out_scan_dir = data_dir+'scans/'
Path(out_scan_dir).mkdir(exist_ok=True)

data_collect=True

if data_collect:
    # MTI connect to RR
    robot_client=MotionProgramExecClient()
    mti_client = RRN.ConnectService("rr+tcp://192.168.55.10:60830/?service=MTI2D")
    mti_client.setExposureTime("25")

    ## path
    q_bp1=[]

    to_start_speed=7
    scan_speed=10

    ## move to start
    q3=[-15,180]
    mp=MotionProgram(ROBOT_CHOICE='ST1',pulse2deg=positioner.pulse2deg)
    mp.MoveJ(q3,10,0)
    robot_client.execute_motion_program(mp)

    mp = MotionProgram(ROBOT_CHOICE='RB2',pulse2deg=robot_scan.pulse2deg)
    mp.MoveJ(np.degrees(q_bp1[0][0]), to_start_speed, 0)
    robot_client.execute_motion_program(mp)

    ## motion start
    mp = MotionProgram(ROBOT_CHOICE='RB2',pulse2deg=robot_scan.pulse2deg)
    # routine motion
    for path_i in range(2,len(q_bp1)-1):
        mp.MoveL(np.degrees(q_bp1[path_i][0]), scan_speed)
    mp.MoveL(np.degrees(q_bp1[-1][0]), scan_speed, 0)

    robot_client.execute_motion_program_nonblocking(mp)
    ###streaming
    robot_client.StartStreaming()
    start_time=time.time()
    state_flag=0
    joint_recording=[]
    robot_stamps=[]
    mti_recording=[]
    r_pulse2deg = np.append(robot_scan.pulse2deg)
    while True:
        if state_flag & 0x08 == 0 and time.time()-start_time>1.:
            break
        res, data = robot_client.receive_from_robot(0.01)
        if res:
            joint_angle=np.radians(np.divide(np.array(data[26:32]),r_pulse2deg))
            state_flag=data[16]
            joint_recording.append(joint_angle)
            timestamp=data[0]+data[1]*1e-9
            robot_stamps.append(timestamp)
            ###MTI scans YZ point from tool frame
            mti_recording.append(deepcopy(np.array([mti_client.lineProfile.X_data,mti_client.lineProfile.Z_data])))
    robot_client.servoMH(False)

    scan_to_home_st = time.time()
    mti_recording=np.array(mti_recording)
    q_out_exe=joint_recording

    # input("Press Enter to Move Home")
    # move robot to home
    q2=np.zeros(6)
    q2[0]=90
    mp=MotionProgram(ROBOT_CHOICE='RB2',pulse2deg=robot_scan.pulse2deg)
    mp.MoveJ(q2,to_start_speed,0)
    robot_client.execute_motion_program(mp)
    #####################
    # exit()

    print("Total exe len:",len(q_out_exe))
    ## save traj
    # save poses
    np.savetxt(out_scan_dir + 'scan_js_exe.csv',q_out_exe,delimiter=',')
    np.savetxt(out_scan_dir + 'scan_robot_stamps.csv',robot_stamps,delimiter=',')
    with open(out_scan_dir + 'mti_scans.pickle', 'wb') as file:
        pickle.dump(mti_recording, file)
    print('Total scans:',len(mti_recording))

q_out_exe=np.loadtxt(out_scan_dir + 'scan_js_exe.csv',delimiter=',')
robot_stamps=np.loadtxt(out_scan_dir +'scan_robot_stamps.csv',delimiter=',')
with open(out_scan_dir+ 'mti_scans.pickle', 'rb') as file:
    mti_recording=pickle.load(file)

scan_process = ScanProcess(robot_scan,positioner)
q_init_table=np.radians([-15,180])
pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,static_positioner_q=q_init_table)
visualize_pcd([pcd])
o3d.io.write_point_cloud(out_scan_dir+'raw_pcd.pcd',pcd)
pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
                                    min_bound=(-1e5,-1e5,10),max_bound=(-1e5,-1e5,30),cluster_based_outlier_remove=True,cluster_neighbor=0.75,min_points=50)
visualize_pcd([pcd])
o3d.io.write_point_cloud(out_scan_dir+'processed_pcd.pcd',pcd)