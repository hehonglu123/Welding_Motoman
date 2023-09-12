from copy import deepcopy
from pathlib import Path
import pickle
import sys
sys.path.append('../toolbox/')
sys.path.append('../scan/scan_tools/')
sys.path.append('../scan/scan_plan/')
sys.path.append('../scan/scan_process/')
from robot_def import *
from scan_utils import *
from scan_continuous import *
from scanPathGen import *
from scanProcess import *

from general_robotics_toolbox import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import glob
import yaml
from math import ceil,floor

zero_config=np.zeros(6)
# 0. robots.
config_dir='../config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'torch.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA1440_marker_config.yaml')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config.yaml')

Table_home_T = positioner.fwd(np.radians([-15,180]))
T_S1TCP_R1Base = np.linalg.inv(np.matmul(positioner.base_H,H_from_RT(Table_home_T.R,Table_home_T.p)))
T_R1Base_S1TCP = np.linalg.inv(T_S1TCP_R1Base)

#### change limit
robot_scan.upper_limit[4]=np.radians(82)
robot_scan.robot.joint_upper_limit[4]=np.radians(82)
robot_scan.lower_limit[4]=np.radians(-82)
robot_scan.robot.joint_lower_limit[4]=np.radians(-82)
positioner.upper_limit[4]=np.radians(-15+0.1)
positioner.robot.joint_upper_limit[4]=np.radians(-15+0.1)
positioner.lower_limit[4]=np.radians(-15-0.1)
positioner.robot.joint_lower_limit[4]=np.radians(-15-0.1)

#### change base H to calibrated ones ####
# robot_scan.base_H = H_from_RT(robot_scan.T_base_basemarker.R,robot_scan.T_base_basemarker.p)
# positioner.base_H = H_from_RT(positioner.T_base_basemarker.R,positioner.T_base_basemarker.p)
# T_to_base = Transform(np.eye(3),[0,0,-380])
# positioner.base_H = np.matmul(positioner.base_H,H_from_RT(T_to_base.R,T_to_base.p))

#### data directory
# dataset='cup/'
# sliced_alg='circular_slice_shifted/'
dataset='sine_wave/'
sliced_alg='auto_slice/'
curve_data_dir = '../data/'+dataset+sliced_alg
scan_data_dir = '../data/'+dataset+sliced_alg+'curve_scan_js/'
scan_p_data_dir = '../data/'+dataset+sliced_alg+'curve_scan_relative/'

#### welding spec, goal
with open(curve_data_dir+'slicing.yml', 'r') as file:
    slicing_meta = yaml.safe_load(file)
line_resolution = slicing_meta['line_resolution']

#### scanning parameters
scan_speed=10 # scanning speed (mm/sec)
scan_stand_off_d = 95 ## mm
Rz_angle = np.radians(0) # point direction w.r.t welds
Ry_angle = np.radians(0) # rotate in y a bit
bounds_theta = np.radians(1) ## circular motion at start and end
extension = 10 ## extension before and after (mm)
all_scan_angle = np.radians([0]) ## scan angle
q_init_table=np.radians([-15,200]) ## init table
R1_w=0.01 ## regularization weight for two robots (R1)
R2_w=0.01 ## regularization weight for two robots (R2)
mti_Rpath = np.array([[ -1.,0.,0.],   
                    [ 0.,1.,0.],
                    [0.,0.,-1.]])

# ## baselayer
# for i in range(0,slicing_meta['num_baselayers']):
#     num_sections=len(glob.glob(curve_data_dir+'curve_sliced_relative/baselayer'+str(i)+'_*.csv'))

#     for x in range(num_sections):
#         print('Base',i,',',x)
#         curve_sliced_relative = np.loadtxt(curve_data_dir+'curve_sliced_relative/baselayer'+str(i)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
#         positioner_weld_js = np.loadtxt(curve_data_dir+'curve_sliced_js/D500B_base_js'+str(i)+'_'+str(x)+'.csv',delimiter=',')

#         curve_sliced_relative=curve_sliced_relative[::-1]
#         positioner_weld_js=positioner_weld_js[::-1]

#         if len(curve_sliced_relative)<2:
#             continue

#         ### scanning path module
#         spg = ScanPathGen(robot_scan,positioner,scan_stand_off_d,Rz_angle,Ry_angle,bounds_theta,extension)
#         # generate scan path
#         print("q init table:",np.degrees(positioner_weld_js[0]))
#         scan_p,scan_R,q_out1,q_out2=spg.gen_scan_path([curve_sliced_relative],[0],all_scan_angle,\
#                             solve_js_method=1,q_init_table=positioner_weld_js[0],R_path=mti_Rpath,R1_w=R1_w,R2_w=R2_w,scan_path_dir=None)
        
#         curve_scan_relative=[]
#         for path_i in range(len(scan_p)):
#             this_T = np.append(scan_p[path_i],R2q(scan_R[path_i]))
#             curve_scan_relative.append(this_T)
        
#         # if i%10==1:
#         #     plt.plot(np.degrees(np.hstack((q_out1,q_out2))),'-o')
#         #     plt.legend(['J1','J2','J3','J4','J5','J6','P1','P2'])
#         #     plt.show()
#         q_out1=np.array(q_out1)
#         q_out2=np.array(q_out2)
#         Path(scan_data_dir).mkdir(exist_ok=True)
#         Path(scan_p_data_dir).mkdir(exist_ok=True)
#         np.savetxt(scan_data_dir+'MA1440_base_js'+str(i)+'_'+str(x)+'.csv',q_out1,delimiter=',')
#         np.savetxt(scan_data_dir+'D500B_base_js'+str(i)+'_'+str(x)+'.csv',q_out2,delimiter=',')
#         np.savetxt(scan_p_data_dir+'scan_base_T'+str(i)+'_'+str(x)+'.csv',curve_scan_relative,delimiter=',')

# exit()

for i in range(0,slicing_meta['num_layers']):
    num_sections=len(glob.glob(curve_data_dir+'curve_sliced_relative/slice'+str(i)+'_*.csv'))
    
    for x in range(num_sections):
        print(i,',',x)
        curve_sliced_relative = np.loadtxt(curve_data_dir+'curve_sliced_relative/slice'+str(i)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
        positioner_weld_js = np.loadtxt(curve_data_dir+'curve_sliced_js/D500B_js'+str(i)+'_'+str(x)+'.csv',delimiter=',')

        curve_sliced_relative=curve_sliced_relative[::-1]
        positioner_weld_js=positioner_weld_js[::-1]

        if len(curve_sliced_relative)<2:
            continue

        ### scanning path module
        spg = ScanPathGen(robot_scan,positioner,scan_stand_off_d,Rz_angle,Ry_angle,bounds_theta,extension)
        # generate scan path
        print("q init table:",np.degrees(positioner_weld_js[0]))
        scan_p,scan_R,q_out1,q_out2=spg.gen_scan_path([curve_sliced_relative],[0],all_scan_angle,\
                            solve_js_method=1,q_init_table=positioner_weld_js[0],R_path=mti_Rpath,R1_w=R1_w,R2_w=R2_w,scan_path_dir=None)
        
        curve_scan_relative=[]
        for path_i in range(len(scan_p)):
            this_T = np.append(scan_p[path_i],R2q(scan_R[path_i]))
            curve_scan_relative.append(this_T)
        
        # if i%10==1:
        #     plt.plot(np.degrees(np.hstack((q_out1,q_out2))),'-o')
        #     plt.legend(['J1','J2','J3','J4','J5','J6','P1','P2'])
        #     plt.show()
        q_out1=np.array(q_out1)
        q_out2=np.array(q_out2)
        Path(scan_data_dir).mkdir(exist_ok=True)
        Path(scan_p_data_dir).mkdir(exist_ok=True)
        np.savetxt(scan_data_dir+'MA1440_js'+str(i)+'_'+str(x)+'.csv',q_out1,delimiter=',')
        np.savetxt(scan_data_dir+'D500B_js'+str(i)+'_'+str(x)+'.csv',q_out2,delimiter=',')
        np.savetxt(scan_p_data_dir+'scan_T'+str(i)+'_'+str(x)+'.csv',curve_scan_relative,delimiter=',')

        