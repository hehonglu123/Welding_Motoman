import sys
sys.path.append('../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *

import cv2
import matplotlib.pyplot as plt
import time

robot=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/scanner_tcp2.csv',\
	pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv')

positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
	pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose_mocap.csv')

q1=np.array([-15,0])
q2=np.array([-15,90])
q3=np.array([-15,190])
q4=np.array([43.5893,72.1362,45.2749,-84.0966,24.3644,94.2091])
q5=np.array([34.6291,55.5756,15.4033,-28.8363,24.0298,3.6855])
q6=np.array([27.3821,51.3582,-19.8428,-21.2525,71.6314,-62.8669])


mp=MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot.pulse2deg,pulse2deg_2=positioner.pulse2deg)
client=MotionProgramExecClient()

target2_1=['MOVJ',q1,1,0]
target2_2=['MOVJ',q1,q2,q3,1,0]
mp.MoveJ(q4,1,0,target2=target2_1)
mp.MoveC(q4, q5, q6, 5,1,target2=target2_2)


client.execute_motion_program(mp)
