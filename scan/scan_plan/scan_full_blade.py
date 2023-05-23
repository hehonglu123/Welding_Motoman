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
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv')
turn_table=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv')


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

### scan parameters
scan_speed=30 # scanning speed (mm/sec)
scan_stand_off_d = 243 ## mm
Rz_angle = np.radians(0) # point direction w.r.t welds
Ry_angle = np.radians(0) # rotate in y a bit, z-axis not pointing down, to have ik solution
# Rz_angle = np.radians(0) # point direction w.r.t welds
# Ry_angle = np.radians(0) # rotate in y a bit, z-axis not pointing down, to have ik solution

bounds_theta = np.radians(5) ## circular motion at start and end

## scan angle
all_scan_angle = np.radians([-45,45]) ## scanning angless

######### enter your wanted layers ########
all_layer=np.arange(0,len(all_curve_sliced_relative),251) ## all layer
if all_layer[-1]!=len(all_curve_sliced_relative)-1:
    all_layer = np.append(all_layer,len(all_curve_sliced_relative)-1)

# param set 1
out_dir = data_dir+''

# path gen
spg = ScanPathGen(robot_scan,turn_table,scan_stand_off_d,Rz_angle,Ry_angle,bounds_theta)
q_init_table=np.radians([15,70])
# q_init_table=np.radians([-15,180])
# q_init_table=np.radians([-70,150])

scan_p,scan_R,q_out1,q_out2=spg.gen_scan_path(all_curve_sliced_relative,all_layer,all_scan_angle,\
                  solve_js_method=1,q_init_table=q_init_table,scan_path_dir=None)

print(np.degrees(q_out1[:10]))
print(np.degrees(q_out1[10:]))

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