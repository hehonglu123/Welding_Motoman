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
    base_marker_config_file=config_dir+'MA2010_0524_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_0524_marker_config.yaml')
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA1440_0524_marker_config.yaml')

positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'D500B_0524_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config.yaml')

#### change base H to calibrated ones ####
robot_scan_base = robot_weld.T_base_basemarker.inv()*robot_scan.T_base_basemarker
robot_scan.base_H = H_from_RT(robot_scan_base.R,robot_scan_base.p)
positioner_base = robot_weld.T_base_basemarker.inv()*positioner.T_base_basemarker
positioner.base_H = H_from_RT(positioner_base.R,positioner_base.p)
T_to_base = Transform(np.eye(3),[0,0,-380])
positioner.base_H = np.matmul(positioner.base_H,H_from_RT(T_to_base.R,T_to_base.p))
# input(positioner.base_H)

robot_weld.robot.P=deepcopy(robot_weld.calib_P)
robot_weld.robot.H=deepcopy(robot_weld.calib_H)
robot_weld.robot.T_flange = deepcopy(robot_weld.T_tool_flange)
robot_weld.robot.R_tool = deepcopy(robot_weld.T_tool_toolmarker.R)
robot_weld.robot.p_tool = deepcopy(robot_weld.T_tool_toolmarker.p)

robot_scan.robot.P=deepcopy(robot_scan.calib_P)
robot_scan.robot.H=deepcopy(robot_scan.calib_H)

#### data directory
# dataset='cup/'
# sliced_alg='circular_slice_shifted/'
dataset='blade0.1/'
sliced_alg='auto_slice/'
curve_data_dir = '../data/'+dataset+sliced_alg

current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-7]
# data_dir=curve_data_dir+'weld_scan_'+formatted_time+'/'
# data_dir=curve_data_dir+'weld_scan_'+'2023_07_11_16_25_30'+'/'
data_dir=curve_data_dir+'weld_scan_'+'2023_07_19_11_41_30'+'/'

# baselayer=False
# layer=367
# x=0

baselayer=True
layer=1
x=0

use_actual = True

if not baselayer:
    layer_data_dir=data_dir+'layer_'+str(layer)+'_'+str(x)+'/'
else:
    layer_data_dir=data_dir+'baselayer_'+str(layer)+'_'+str(x)+'/'
out_scan_dir = layer_data_dir+'scans/'

if not baselayer:
    curve_sliced_relative=np.loadtxt(curve_data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
else:
    curve_sliced_relative=np.loadtxt(curve_data_dir+'curve_sliced_relative/baselayer'+str(layer)+'_'+str(x)+'.csv',delimiter=',')

### robot 1
weld_q_exe = np.loadtxt(layer_data_dir+'weld_js_exe.csv',delimiter=',')
if use_actual:
    curve_sliced_relative=[]
    for q_out in weld_q_exe[::-1]:
        Table_home_T = positioner.fwd(q_out[-2:])
        T_S1TCP_R1Base = np.matmul(positioner.base_H,H_from_RT(Table_home_T.R,Table_home_T.p))
        T_R1Base_S1TCP = np.linalg.inv(T_S1TCP_R1Base)
        robot_T = robot_weld.fwd(q_out[:6])
        T_R1TCP_S1TCP = np.matmul(T_R1Base_S1TCP,H_from_RT(robot_T.R,robot_T.p))
        # print(robot_T)
        # input(T_R1TCP_S1TCP[:3,-1])
        if len(curve_sliced_relative)==0 or np.linalg.norm(T_R1TCP_S1TCP[:3,-1]-curve_sliced_relative[-1][:3])>0.8:
            if len(curve_sliced_relative)==0 or np.fabs(T_R1TCP_S1TCP[2,-1]-curve_sliced_relative[-1][2])<0.4:
                curve_sliced_relative.append(np.append(T_R1TCP_S1TCP[:3,-1],T_R1TCP_S1TCP[:3,2]))
    curve_sliced_relative=curve_sliced_relative[::-1]
    print("curve len:",len(curve_sliced_relative))
###########

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
pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
                                    min_bound=crop_min,max_bound=crop_max,outlier_remove=True,cluster_based_outlier_remove=False,cluster_neighbor=1,min_points=25)
visualize_pcd([pcd])
profile_height = scan_process.pcd2dh(pcd,curve_sliced_relative,drawing=True)

curve_i=0
total_curve_i = len(profile_height)
for curve_i in range(total_curve_i):
    color_dist = plt.get_cmap("rainbow")(float(curve_i)/total_curve_i)
    plt.scatter(profile_height[curve_i,0],profile_height[curve_i,1],c=color_dist)
plt.xlabel('Lambda')
plt.ylabel('dh to Layer N (mm)')
plt.title("Height Profile")
plt.show()