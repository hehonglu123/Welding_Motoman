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
import numpy as np

def robot_weld_path_gen(all_layer_z):
    R=np.array([[ 0.7071, -0.7071, -0.    ],
			[-0.7071, -0.7071,  0.    ],
			[-0.,      0.,     -1.    ]])
    # x0 =  1684	# Origin x coordinate
    # y0 = -753.5	# Origin y coordinate
    # z0 = -245   # 10 mm distance to base

    # base layer
    # weld_p = np.array([[1651, -771, -245],[1651, -856, -245]])
    # wall layer
    weld_p = np.array([[1651, -781, -245],[1651, -846, -245]])

    ## tune
    dx = 0
    dy = 0
    dz = 0 # observe z height different (posibiliy from calib error)
    dp = np.array([dx,dy,dz])

    all_path_T=[]
    for layer_z in all_layer_z:
        path_T=[]
        for p in weld_p:
            path_T.append(Transform(R,p+dp+np.array([0,0,layer_z])))

        all_path_T.append(path_T)
    
    return all_path_T

# 0. robots. Note use "(robot)_pose_mocapcalib.csv"
config_dir='../config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'torch.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
	base_transformation_file=config_dir+'MA1440_pose_mocapcalib.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA1440_marker_config.yaml')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose_mocapcalib.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config.yaml')

Table_home_T = positioner.fwd(np.radians([-15,180]))
T_S1TCP_R1Base = np.linalg.inv(np.matmul(positioner.base_H,H_from_RT(Table_home_T.R,Table_home_T.p)))

# 1. The curve path in "positioner tcp frame"
######### enter your wanted z height #######
all_layer_z = [30]
###########################################
all_path_T = robot_weld_path_gen(all_layer_z) # this is your path
path_T=all_path_T[0]
curve_sliced_relative=[]
for path_p in path_T:
    this_p = np.matmul(T_S1TCP_R1Base[:3,:3],path_p.p)+T_S1TCP_R1Base[:3,-1]
    this_n = np.matmul(T_S1TCP_R1Base[:3,:3],path_p.R[:,-1])
    curve_sliced_relative.append(np.append(this_p,this_n))

# 2. Scanning parameters
### scan parameters
scan_speed=10 # scanning speed (mm/sec)
scan_stand_off_d = 70 ## mm
Rz_angle = np.radians(0) # point direction w.r.t welds
Ry_angle = np.radians(0) # rotate in y a bit
bounds_theta = np.radians(10) ## circular motion at start and end
all_scan_angle = np.radians([0]) ## scan angle
q_init_table=np.radians([-30,0]) ## init table
save_output_points = False
save_points_dir = ''

### scanning path module
spg = ScanPathGen(robot_scan,positioner,scan_stand_off_d,Rz_angle,Ry_angle,bounds_theta)

mti_Rpath = np.array([[ -1.,0.,0.],   
                    [ 0.,1.,0.],
                    [0.,0.,-1.]])
# generate scan path
scan_p,scan_R,q_out1,q_out2=spg.gen_scan_path([curve_sliced_relative],[0],all_scan_angle,\
                  solve_js_method=0,q_init_table=q_init_table,R_path=mti_Rpath,scan_path_dir=None)
# generate motion program
q_bp1,q_bp2,s1_all,s2_all=spg.gen_motion_program(q_out1,q_out2,scan_p,scan_speed,init_sync_move=0)

######################################################
########### Do welding #############
######################################################

######## scanning motion #########
### execute motion ###
robot_client=MotionProgramExecClient()
input("Press Enter to start moving")

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
    out_scan_dir = save_points_dir+'scans/'
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
z_height_start=0
profile_height = scan_process.pcd2height(pcd,z_height_start)