from copy import deepcopy
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

def robot_weld_path_gen(test_n):
    R=np.array([[ 0.7071, -0.7071, -0.    ],
			[-0.7071, -0.7071,  0.    ],
			[-0.,      0.,     -1.    ]])
    x0 =  1684	# Origin x coordinate
    y0 = -1179	# Origin y coordinate
    z0 = -245   # 10 mm distance to base

    all_path_T = []
    for n in test_n:
        p_start = [x0, y0 - 12, z0 - 10]
        p_end = [x0 - 76, y0 - 12 , z0 - 10]

        T_start=Transform(R,p_start)
        T_end = Transform(R,p_end)
        all_path_T.append([T_start,T_end])

        y0 = y0 - 27
    
    return all_path_T

data_dir='test2/'
config_dir='../../config/'

robot_weld=robot_obj('MA_2010_A0',def_path='../../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../../config/weldgun.csv',\
	pulse2deg_file_path='../../config/MA_2010_A0_pulse2deg.csv')
robot_scan=robot_obj('MA_1440_A0',def_path=config_dir+'MA_1440_A0_robot_default_config.yml',tool_file_path=config_dir+'scanner_tcp2.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA_1440_A0_pulse2deg.csv')
turn_table=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',base_transformation_file=config_dir+'D500B_pose.csv',\
    pulse2deg_file_path=config_dir+'D500B_pulse2deg.csv')

### get welding robot path
test_n=4 # how many test
all_path_T = robot_weld_path_gen(test_n)

### scan parameters
scan_stand_off_d = 5 ## mm
all_scan_angle = np.radians([-30,-15,0,15,30]) ## scanning angles

for path_T in all_path_T:

    ### path gen ###
    scan_path=[]
    robot_path=deepcopy(path_T)
    for scan_angle in all_scan_angle:
        for pT in robot_path:
            pass

    ### redundancy resolution ###
    ### motion program generation ###
