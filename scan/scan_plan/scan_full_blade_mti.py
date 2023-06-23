from audioop import reverse
from copy import deepcopy
from pathlib import Path
import sys
sys.path.append('../../toolbox/')
sys.path.append('../scan_tools/')
from robot_def import *
from scan_utils import *
from scan_continuous import *
from scanPathGen import *
from dx200_motion_program_exec_client import *

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time
import numpy as np

config_dir='../../config/'

# robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'torch.csv',\
# 	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
# print(robot_weld.fwd(zero_config))
# exit()
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'scanner_tcp2.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA1440_marker_config.yaml',tool_marker_config_file=config_dir+'scanner_marker_config.yaml')

turn_table=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config.yaml')

#### change base H to calibrated ones ####
robot_scan.base_H = H_from_RT(robot_scan.T_base_basemarker.R,robot_scan.T_base_basemarker.p)
turn_table.base_H = H_from_RT(turn_table.T_base_basemarker.R,turn_table.T_base_basemarker.p)
T_to_base = Transform(np.eye(3),[0,0,-380])
turn_table.base_H = np.matmul(turn_table.base_H,H_from_RT(T_to_base.R,T_to_base.p))

## wall test path
dataset='blade0.1/'
sliced_alg='auto_slice/'
data_dir='../../data/'+dataset+sliced_alg
all_curve_sliced_relative=[]
# import base layer
all_curve_sliced_relative.append(np.loadtxt(data_dir+'curve_sliced_relative/baselayer0_0.csv',delimiter=','))
all_curve_sliced_relative.append(np.loadtxt(data_dir+'curve_sliced_relative/baselayer1_0.csv',delimiter=','))
total_layer=500
for layer in range(total_layer):
    part_num=0
    all_curve_sliced_relative.append([])
    while True:
        try:
            all_curve_sliced_relative[-1].extend(np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(part_num)+'.csv',delimiter=','))
        except:
            break
        part_num+=1
print("Total Layer",len(all_curve_sliced_relative))
print(all_curve_sliced_relative[0][0])

### scan parameters
scan_speed=30 # scanning speed (mm/sec)
scan_stand_off_d = 85 ## mm
Rz_angle = np.radians(0) # point direction w.r.t welds
Ry_angle = np.radians(0) # rotate in y a bit, z-axis not pointing down, to have ik solution
# Rz_angle = np.radians(0) # point direction w.r.t welds
# Ry_angle = np.radians(0) # rotate in y a bit, z-axis not pointing down, to have ik solution

bounds_theta = np.radians(5) ## circular motion at start and end
extension = 10 # mm

## scan angle
all_scan_angle = np.radians([-30,30]) ## scanning angless

######### enter your wanted layers ########
all_layer=np.arange(0,len(all_curve_sliced_relative),251) ## all layer
if all_layer[-1]!=len(all_curve_sliced_relative)-1:
    all_layer = np.append(all_layer,len(all_curve_sliced_relative)-1)

# param set 1
out_dir = data_dir+''

# path gen
spg = ScanPathGen(robot_scan,turn_table,scan_stand_off_d,Rz_angle,Ry_angle,bounds_theta,extension)
q_init_table=np.radians([30,70])
# q_init_table=np.radians([-15,180])
# q_init_table=np.radians([-70,150])

mti_Rpath = np.array([[ -1.,0.,0.],   
                    [ 0.,1.,0.],
                    [0.,0.,-1.]])

scan_p,scan_R,q_out1,q_out2=spg.gen_scan_path(all_curve_sliced_relative,all_layer,all_scan_angle,\
                  solve_js_method=1,q_init_table=q_init_table,R_path=mti_Rpath,scan_path_dir=None)

print(np.degrees(q_out1[:10]))
print(np.degrees(q_out1[10:]))

# motion program gen
q_bp1,q_bp2,s1_all,s2_all=spg.gen_motion_program(q_out1,q_out2,scan_p,scan_speed,init_sync_move=0)

# exit()

### execute motion ###
robot_client=MotionProgramExecClient()
use_artec_studio=False
input("Press Enter to start moving")

## move to start
to_start_speed=3
mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot_scan.pulse2deg,pulse2deg_2=turn_table.pulse2deg)
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
r_pulse2deg = np.append(robot_scan.pulse2deg,turn_table.pulse2deg)
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
        mti_recording.append(np.array([-mti_client.lineProfile.X_data,mti_client.lineProfile.Z_data]))
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
mp=MotionProgram(ROBOT_CHOICE='ST1',pulse2deg=turn_table.pulse2deg)
mp.MoveJ(q3,10,0)
robot_client.execute_motion_program(mp)
#####################
# exit()

print("Total exe len:",len(q_out_exe))

out_scan_dir = out_dir+'scans/'
## save traj
Path(out_scan_dir).mkdir(exist_ok=True)
# save poses
np.savetxt(out_scan_dir + 'scan_js_exe.csv',q_out_exe,delimiter=',')
np.savetxt(out_scan_dir + 'robot_stamps.csv',robot_stamps,delimiter=',')
with open('mti_scans.pickle', 'wb') as file:
    pickle.dump(mti_recording, file)