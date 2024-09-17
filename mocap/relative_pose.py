from copy import deepcopy
import sys
sys.path.append('../toolbox/')
from utils import *
from robot_def import * 

from general_robotics_toolbox import *
# from RobotRaconteur.Client import *
import numpy as np
# from MocapPoseListener import *
import pickle

config_dir='../config/'

R1_dataset_date='06162024'
robot_marker_dir=config_dir+'MA2010_marker_config/'
tool_marker_dir=config_dir+'weldgun_marker_config/'
robot_1=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',\
                    tool_file_path=config_dir+'torch.csv',d=15,\
                    #  tool_file_path='',d=0,\
                    pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
                    base_marker_config_file=robot_marker_dir+'MA2010_'+R1_dataset_date+'_marker_config.yaml',\
                    tool_marker_config_file=tool_marker_dir+'weldgun_'+R1_dataset_date+'_marker_config.yaml')

R2_dataset_date='06162024'
robot_marker_dir=config_dir+'MA1440_marker_config/'
tool_marker_dir=config_dir+'mti_marker_config/'
robot_2=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',\
                    tool_file_path=config_dir+'mti.csv',\
                    pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
                    base_marker_config_file=robot_marker_dir+'MA1440_'+R2_dataset_date+'_marker_config.yaml',\
                    tool_marker_config_file=tool_marker_dir+'mti_'+R2_dataset_date+'_marker_config.yaml')

S1_dataset_date='0926'
base_marker_dir=config_dir+'D500B_marker_config/'
tool_marker_dir=config_dir+'positioner_tcp_marker_config/'
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
base_marker_config_file=base_marker_dir+'D500B_'+S1_dataset_date+'_marker_config.yaml',tool_marker_config_file=tool_marker_dir+'positioner_tcp_'+S1_dataset_date+'_marker_config.yaml')

## Transformations
T_R2_R1 = robot_1.T_base_basemarker.inv()*robot_2.T_base_basemarker
T_S1_R1 = robot_1.T_base_basemarker.inv()*positioner.T_base_basemarker
robot_2.base_H = H_from_RT(T_R2_R1.R,T_R2_R1.p)
positioner.base_H = H_from_RT(T_S1_R1.R,T_S1_R1.p)
T_to_base = Transform(np.eye(3),[0,0,-380])
positioner.base_H = np.matmul(positioner.base_H,H_from_RT(T_to_base.R,T_to_base.p))
print("MA1440:",robot_2.base_H)
print("D500B:",positioner.base_H)
np.savetxt(config_dir+'MA1440_pose_mocapcalib.csv',robot_2.base_H,delimiter=',')
np.savetxt(config_dir+'D500B_pose_mocapcalib.csv',positioner.base_H,delimiter=',')
exit()