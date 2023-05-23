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

def robot_weld_path_gen():
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
    dz = 0
    dp = np.array([dx,dy,dz])

    path_T=[]
    for p in weld_p:
        path_T.append(Transform(R,p+dp))

    all_path_T = [path_T]
    
    return all_path_T

config_dir='../../config/'

# robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'torch.csv',\
# 	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
# print(robot_weld.fwd(zero_config))
# exit()
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'scanner_tcp2.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv')
turn_table=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv')


## wall test path
Table_home_T = turn_table.fwd(np.radians([-15,180]))
T_S1TCP_R1Base = np.linalg.inv(np.matmul(turn_table.base_H,H_from_RT(Table_home_T.R,Table_home_T.p)))

data_dir='../../data/wall_weld_test/top_layer_test/'
all_path_T = robot_weld_path_gen()
curve_sliced_R1=np.array(all_path_T[0])
curve_sliced_relative=[]
for path_p in curve_sliced_R1:
    this_p = np.matmul(T_S1TCP_R1Base[:3,:3],path_p.p)+T_S1TCP_R1Base[:3,-1]
    this_n = np.matmul(T_S1TCP_R1Base[:3,:3],path_p.R[:,-1])
    curve_sliced_relative.append(np.append(this_p,this_n))
curve_sliced_relative=np.array(curve_sliced_relative)

### scan parameters
scan_speed=30 # scanning speed (mm/sec)
scan_stand_off_d = 243 ## mm
Rz_angle = np.radians(-45) # point direction w.r.t welds
Ry_angle = np.radians(0) # rotate in y a bit, z-axis not pointing down, to have ik solution
# Rz_angle = np.radians(0) # point direction w.r.t welds
# Ry_angle = np.radians(0) # rotate in y a bit, z-axis not pointing down, to have ik solution

bounds_theta = np.radians(5) ## circular motion at start and end

## scan angle
all_scan_angle = np.radians([0]) ## scanning angless

######### enter your wanted z height ########
all_layer_z=[0] ## all layer z height

# param set 1
out_dir = data_dir+''

# path gen
spg = ScanPathGen(robot_scan,turn_table,scan_stand_off_d,Rz_angle,Ry_angle,bounds_theta)
q_init_table=np.radians([-40,170])
# q_init_table=np.radians([-15,180])

scan_p,scan_R,q_out1,q_out2=spg.gen_scan_path(curve_sliced_relative,all_layer_z,all_scan_angle,\
                  solve_js_method=0,q_init_table=q_init_table,scan_path_dir=out_dir)

# print(np.degrees(q_out1[:10]))
# print(np.degrees(q_out1[10:]))

# motion program gen
q_bp1,q_bp2,s1_all,s2_all=spg.gen_motion_program(q_out1,q_out2,scan_p,scan_speed)

exit()

### execute motion ###
robot_client=MotionProgramExecClient(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot_scan.pulse2deg,pulse2deg_2=turn_table.pulse2deg)
use_artec_studio=False
input("Press Enter to start moving")

## move to start
to_start_speed=10
mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot_scan.pulse2deg,pulse2deg_2=turn_table.pulse2deg)
target2=['MOVJ',np.degrees(q_bp2[0][0]),to_start_speed]
mp.MoveJ(np.degrees(q_bp1[0][0]), to_start_speed, 0, target2=target2)
robot_client.execute_motion_program(mp)

input("Open Artec Studio or Scanner and Press Enter to start moving")

if not use_artec_studio:
    ## scanner start
    ### scanner hardware
    c = RRN.ConnectService('rr+tcp://localhost:64238?service=scanner')
    cscanner = ContinuousScanner(c)
    cscanner.start_capture()

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
robot_stamps,curve_pulse_exe,_,_ = robot_client.execute_motion_program(mp)
q_out_exe=curve_pulse_exe[:,6:]

if not use_artec_studio:
    ## scanner end
    cscanner.end_capture()
    scans,scan_stamps=cscanner.get_capture()

input("Press Stop on Artec Studio and Move Home")
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
if not use_artec_studio:
    out_scan_dir = out_dir+'scans/'
    ## save traj
    Path(out_scan_dir).mkdir(exist_ok=True)
    # save poses
    np.savetxt(out_scan_dir + 'scan_js_exe.csv',q_out_exe,delimiter=',')
    np.savetxt(out_scan_dir + 'robot_stamps.csv',robot_stamps,delimiter=',')
    scan_count=0
    for scan in scans:
        scan_points = RRN.NamedArrayToArray(scan.vertices)
        np.save(out_scan_dir + 'points_'+str(scan_count)+'.npy',scan_points)
        if scan_count%10==0:
            print("scan counts:",len(scan_points))
        scan_count+=1
    print('Total scans:',scan_count)
    np.savetxt(out_scan_dir + 'scan_stamps.csv',scan_stamps,delimiter=',')