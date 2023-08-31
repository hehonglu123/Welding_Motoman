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
q3=np.array([-15,180])
q1_1=np.array([-15,360])
q2_1=np.array([-15,270])

q4=np.array([35.521,65.7516,21.1823,-78.2675,33.4785,100.7489])
q5=np.array([26.1197,32.5243,-25.0171,-19.9194,47.7077,12.1999])
q6=np.array([7.78,39.94,-46.4878,-13.8057,81.5178,-62.1720])


mp=MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot.pulse2deg,pulse2deg_2=positioner.pulse2deg)
client=MotionProgramExecClient()

target2_1=['MOVJ',q1,1,0]
target2_2=['MOVJ',q1,q2,q3,1,0]
mp.MoveJ(q4,10,0,target2=target2_1)
mp.MoveC(q4, q5, q6, 10,1,target2=target2_2)
mp.MoveJ(q6,10,0,target2=['MOVJ',q3,1,0])
target2_2=['MOVJ',q3,q2_1,q1_1,1,0]
mp.MoveC(q6, q5, q4, 10,1,target2=target2_2)

client.execute_motion_program(mp)
