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
from weldCorrectionStrategy import *

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

#### change base H to calibrated ones ####
robot_scan.base_H = H_from_RT(robot_scan.T_base_basemarker.R,robot_scan.T_base_basemarker.p)
positioner.base_H = H_from_RT(positioner.T_base_basemarker.R,positioner.T_base_basemarker.p)
T_to_base = Transform(np.eye(3),[0,0,-380])
positioner.base_H = np.matmul(positioner.base_H,H_from_RT(T_to_base.R,T_to_base.p))

#### data directory
dataset='cup/'
sliced_alg='circular_slice_shifted/'
curve_data_dir = '../data/'+dataset+sliced_alg

current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-7]
# data_dir=curve_data_dir+'weld_scan_'+formatted_time+'/'
data_dir=curve_data_dir+'weld_scan_'+'2023_07_10_16_59_28'+'/'


layer=0
x=40
layer_data_dir=data_dir+'layer_'+str(layer)+'_'+str(x)+'/'
out_scan_dir = layer_data_dir+'scans/'
curve_sliced_relative=np.loadtxt(curve_data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')

with open(out_scan_dir+'mti_scans.pickle', 'rb') as file:
    mti_recording=pickle.load(file)
q_out_exe=np.loadtxt(out_scan_dir+'scan_js_exe.csv',delimiter=',')
robot_stamps=np.loadtxt(out_scan_dir+'scan_robot_stamps.csv',delimiter=',')

#### scanning process: processing point cloud and get h
curve_sliced_relative=np.array(curve_sliced_relative)
crop_extend=15
crop_min=tuple(np.min(curve_sliced_relative[:,:3],axis=0)-crop_extend)
crop_max=tuple(np.max(curve_sliced_relative[:,:3],axis=0)+crop_extend)
print(crop_min)
print(crop_max)
scan_process = ScanProcess(robot_scan,positioner)
pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps)
# visualize_pcd([pcd])
pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=20,std_ratio=1.5,\
                                    min_bound=crop_min,max_bound=crop_max,outlier_remove=False,cluster_based_outlier_remove=False,cluster_neighbor=1,min_points=25)
visualize_pcd([pcd])
profile_height = scan_process.pcd2dh(pcd,curve_sliced_relative)
plt.scatter(profile_height[:,0],profile_height[:,1])
plt.show()