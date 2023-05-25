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

mocap_filename='PH_raw_data/test0509_S1_aftercalib/train_data_zero_mocap'
with open(mocap_filename+'_p.pickle', 'rb') as handle:
    curve_p = pickle.load(handle)
with open(mocap_filename+'_R.pickle', 'rb') as handle:
    curve_R = pickle.load(handle)
with open(mocap_filename+'_timestamps.pickle', 'rb') as handle:
    mocap_stamps = pickle.load(handle)

T_basemarker = Transform(curve_R[positioner.base_rigid_id][0],curve_p[positioner.base_rigid_id][0])
T_toolmarker = Transform(curve_R[positioner.tool_rigid_id][0],curve_p[positioner.tool_rigid_id][0])

print(positioner.T_base_basemarker.inv()*T_basemarker.inv()*T_toolmarker*positioner.T_tool_toolmarker)

## Transformations
# T_R2_R1 = robot_weld.T_base_basemarker.inv()*robot_scan.T_base_basemarker
# T_S1_R1 = robot_weld.T_base_basemarker.inv()*turn_table.T_base_basemarker
# robot_scan.base_H = H_from_RT(T_R2_R1.R,T_R2_R1.p)
# turn_table.base_H = H_from_RT(T_S1_R1.R,T_S1_R1.p)
# T_to_base = Transform(np.eye(3),[0,0,-380])
# turn_table.base_H = np.matmul(turn_table.base_H,H_from_RT(T_to_base.R,T_to_base.p))
# np.savetxt(config_dir+'MA1440_pose_mocapcalib.csv',robot_scan.base_H,delimiter=',')
# np.savetxt(config_dir+'D500B_pose_mocapcalib.csv',turn_table.base_H,delimiter=',')
# exit()