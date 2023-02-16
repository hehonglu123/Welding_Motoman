import sys
sys.path.append('../../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *
from MotionSendMotoman import *
from general_robotics_toolbox import *

from RobotRaconteur.Client import *
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np
from math import ceil

data_dir='../../data/wall_weld_test/static_wall_test1'
config_dir='../../config/'

scan_resolution=10 #scan every 5 mm

## kinematics definition
robot=robot_obj('MA_1440_A0',def_path=config_dir+'MA_1440_A0_robot_default_config.yml',tool_file_path=config_dir+'scanner_tcp2.csv',\
	pulse2deg_file_path=config_dir+'MA_1440_A0_pulse2deg.csv')
turn_table=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',base_transformation_file=config_dir+'D500B_pose.csv',\
    pulse2deg_file_path=config_dir+'D500B_pulse2deg.csv')

## clients for hardwares
scanner_client=RRN.ConnectService('rr+tcp://192.168.55.27:64238?service=scanner')
robot_client=MotionProgramExecClient(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot.pulse2deg,pulse2deg_2=turn_table.pulse2deg)

# the path
# q1=np.array([43.9704, 69.2684, 39.7750, -78.7448, 25.0877, 88.1245])
q2=np.array([28.7058, 52.7440, 0.3853, 58.5666, 89.9525, -30.5505])
q3=np.array([32.5383, 52.5302, 6.9630, 65.8277, 70.3430, -37.1343])
q4=np.array([34.7476, 55.2934, 12.0979, 64.8555, 67.9181, -38.7352])
q5=np.array([37.9175, 58.6948, 21.4432, 75.3962, 54.2387, -49.0660])

q6=np.array([41.2526, 67.3697, 36.1898, 91.7557, 44.0331, -65.4353])
q7=np.array([38.5487, 60.8028, 23.8661, 87.5627, 46.3131, -57.3386])
q8=np.array([31.9357, 54.9688, 6.2205, 58.3067, 82.7573, -32.7165])
q9=np.array([26.9181, 50.9167, -2.2788, 61.2708, 89.0354, -30.4098])

t1=[-15,180]
t2=[-15,220]
t3=[-15,260]
t4=[-15,300]
t5=[-15,340]

robot_path=[q6,q7,q8,q9,q2,q3,q4,q5,q4,q3,q2,q4,q2,q6,q7,q8,q9,q2,q3,q4,q5,q3]
table_path=[t1,t1,t1,t1,t1,t1,t1,t1,t1,t2,t3,t4,t5,t5,t5,t5,t5,t5,t5,t5,t5,t1]

# for path_i in range(len(robot_path)-1):
#     print(np.linalg.norm(robot.fwd(np.radians(robot_path[path_i])).p-robot.fwd(np.radians(robot_path[path_i+1])).p))
# exit()

scan_count=0
joint_scan_poses=[]

## move to start
robot_client.MoveJ(q6, 5, 0, target2=['MOVJ',[-15,180],3,0])
robot_client.ProgEnd()
timestamps, joint_recording=robot_client.execute_motion_program("AAA.JBI")
print('Current pose:',np.degrees(joint_recording[-1]), ',Scan Count:',scan_count)
mesh=scanner_client.capture(True)
scan_points = RRN.NamedArrayToArray(mesh.vertices)
np.savetxt(data_dir + 'points_'+str(scan_count)+'.csv',scan_points,delimiter=',')
joint_scan_poses.append(joint_recording[-1])
scan_count+=1

for path_i in range(1,len(robot_path)):
    path_length=np.linalg.norm(robot.fwd(np.radians(robot_path[path_i])).p-robot.fwd(np.radians(robot_path[path_i-1])).p)
    path_steps = ceil(path_length/scan_resolution)

    this_robot_path = np.linspace(robot_path[path_i-1],robot_path[path_i],path_steps)
    this_table_path = np.linspace(table_path[path_i-1],table_path[path_i],path_steps)
    for this_path_i in range(1,path_steps):
        # print(this_robot_path[this_path_i],this_table_path[this_path_i])
        # time.sleep(0.5)
        robot_client = MotionProgramExecClient(ROBOT_CHOICE='RB2', ROBOT_CHOICE2='ST1', pulse2deg=robot.pulse2deg,
                                               pulse2deg_2=turn_table.pulse2deg)

        robot_client.MoveJ(this_robot_path[this_path_i], 5, 0, target2=['MOVJ',this_table_path[this_path_i],3,0])
        robot_client.ProgEnd()
        timestamps, joint_recording=robot_client.execute_motion_program("AAA.JBI")
        print('Current pose:',np.degrees(joint_recording[-1]), ',Scan Count:',scan_count)
        mesh=scanner_client.capture(True)
        scan_points = RRN.NamedArrayToArray(mesh.vertices)
        np.savetxt(data_dir + 'points_'+str(scan_count)+'.csv',scan_points,delimiter=',')
        joint_scan_poses.append(joint_recording[-1])
        scan_count+=1
    print("change target")

# save poses
np.savetxt(data_dir + 'scan_js_exe.csv',joint_scan_poses,delimiter=',')