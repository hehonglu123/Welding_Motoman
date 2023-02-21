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

def get_bound_circle(p,R,pc,k,theta):

    p=np.array(p)
    R=np.array(R)
    pc=np.array(pc)
    k=np.array(k)
    
    rot_R_2 = rot(k,theta/2)
    rot_R = rot(k,theta)
    p_bound = np.matmul(rot_R,(p-pc))-(p-pc)+pc
    p_bound_2 = np.matmul(rot_R_2,(p-pc))-(p-pc)+pc
    R_bound = np.matmul(rot_R,R)
    R_bound_2 = np.matmul(rot_R_2,R)
    
    return [p_bound,p_bound_2],[R_bound,R_bound_2]

data_dir='test2/'
config_dir='../../config/'

robot_weld=robot_obj('MA_2010_A0',def_path='../../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../../config/weldgun.csv',\
	pulse2deg_file_path='../../config/MA_2010_A0_pulse2deg.csv')
robot_scan=robot_obj('MA_1440_A0',def_path=config_dir+'MA_1440_A0_robot_default_config.yml',tool_file_path=config_dir+'scanner_tcp2.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA_1440_A0_pulse2deg.csv')
turn_table=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',base_transformation_file=config_dir+'D500B_pose.csv',\
    pulse2deg_file_path=config_dir+'D500B_pulse2deg.csv')

R2base_R1base_H = np.linalg.inv(robot_scan.base_H)

### get welding robot path
test_n=4 # how many test
all_path_T = robot_weld_path_gen(test_n)

### scan parameters
scan_stand_off_d = 5 ## mm
bounds_theta = np.radians(45) ## circular motion at start and end
all_scan_angle = np.radians([-30,-15,0,15,30]) ## scanning angles

for path_T in all_path_T:

    ### path gen ###
    scan_path=[]
    robot_path=deepcopy(path_T)
    for scan_angle in all_scan_angle:
        scan_p=[]
        scan_R=[]
        for pT_i in range(len(robot_path)):

            this_p = robot_path[pT_i].p
            this_R = robot_path[pT_i].R

            if pT_i == len(robot_path)-1:
                this_scan_R = deepcopy(scan_R[-1])
                this_scan_p = this_p - this_scan_R[:,-1]*scan_stand_off_d # stand off distance to scan
                scan_p.append(this_scan_p)
                scan_R.append(this_scan_R)
                k = deepcopy(this_scan_R[:,1])
                p_bound_path,R_bound_path=get_bound_circle(this_scan_p,this_scan_R,this_p,k,bounds_theta)
                scan_p.extend(p_bound_path[::-1])
                scan_R.extend(R_bound_path[::-1])
                break

            next_p = robot_path[pT_i].p
            next_r = robot_path[pT_i].R
            travel_v = (next_p-this_p)
            travel_v = travel_v/np.linalg.norm(travel_v)

            # get scan R
            Rx = travel_v
            Rz = this_R[:,-1] # assume weld is perpendicular to the plane
            Rz = (Rz-np.dot(Rx,Rz)*Rz)
            Rz = Rz/np.linalg.norm(Rz)
            Ry = np.cross(Rz,Rx)
            Ry = Ry/np.linalg.norm(Ry)
            this_scan_R = np.array([Rx,Ry,Rz]).T
            # get scan p
            this_scan_p = this_p - Rz*scan_stand_off_d # stand off distance to scan

            # add start bound condition
            if pT_i == 0:
                k = deepcopy(Ry)
                p_bound_path,R_bound_path=get_bound_circle(this_scan_p,this_scan_R,this_p,k,bounds_theta)
                scan_p.extend(p_bound_path)
                scan_R.extend(R_bound_path)
            
            # add scan p R to path
            scan_p.append(this_scan_p)
            scan_R.append(this_scan_R)

    ### change everything to appropriate frame (currently: Robot2):
    scan_p = np.matmul()

    ### redundancy resolution ###
    ### motion program generation ###
