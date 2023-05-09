from copy import deepcopy
import sys
sys.path.append('../toolbox/')
from utils import *
from robot_def import * 

from general_robotics_toolbox import *
# from RobotRaconteur.Client import *
import numpy as np
# from MocapPoseListener import *

config_dir='../config/'
base_marker_config_file=config_dir+'MA2010_marker_config.yaml'
tool_marker_config_file=config_dir+'weldgun_marker_config.yaml'
robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
base_marker_config_file=base_marker_config_file,tool_marker_config_file=tool_marker_config_file)

base_marker_config_file=config_dir+'D500B_marker_config.yaml'
tool_marker_config_file=config_dir+'positioner_tcp_marker_config.yaml'
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
base_marker_config_file=base_marker_config_file,tool_marker_config_file=tool_marker_config_file)

# positioner transform to bottom
pos_T_bottombase_basemarker = positioner.T_base_basemarker*Transform(np.eye(3),-1*(positioner.robot.P[:,0]))
# pos_T_bottombase_basemarker = positioner.T_base_basemarker*Transform(np.eye(3),positioner.robot.H[:,1]*(positioner.robot.P[2,0]))

# T^positioner base _ robot base
T_posbase_robbase = robot.T_base_basemarker.inv()*pos_T_bottombase_basemarker

print(T_posbase_robbase)
print(R2rpy(T_posbase_robbase.R))
print(T_posbase_robbase.p)