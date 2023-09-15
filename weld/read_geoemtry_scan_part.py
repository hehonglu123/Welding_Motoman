from copy import deepcopy
from pathlib import Path
import pickle
import sys
sys.path.append('../toolbox/')
sys.path.append('../scan/scan_tools/')
sys.path.append('../scan/scan_plan/')
sys.path.append('../scan/scan_process/')
sys.path.append('../mocap/')
from robot_def import *
from scan_utils import *
from scan_continuous import *
from scanPathGen import *
from scanProcess import *
from PH_interp import *
from weldCorrectionStrategy import *

from general_robotics_toolbox import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import glob
import yaml
from math import ceil,floor

R1_ph_dataset_date='0801'
R2_ph_dataset_date='0801'
S1_ph_dataset_date='0801'

zero_config=np.zeros(6)
# 0. robots.
config_dir='../config/'
R1_marker_dir=config_dir+'MA2010_marker_config/'
weldgun_marker_dir=config_dir+'weldgun_marker_config/'
R2_marker_dir=config_dir+'MA1440_marker_config/'
mti_marker_dir=config_dir+'mti_marker_config/'
S1_marker_dir=config_dir+'D500B_marker_config/'
S1_tcp_marker_dir=config_dir+'positioner_tcp_marker_config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'torch.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
    base_marker_config_file=R1_marker_dir+'MA2010_'+R1_ph_dataset_date+'_marker_config.yaml',tool_marker_config_file=weldgun_marker_dir+'weldgun_'+R1_ph_dataset_date+'_marker_config.yaml')
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
    base_marker_config_file=R2_marker_dir+'MA1440_'+R2_ph_dataset_date+'_marker_config.yaml',tool_marker_config_file=mti_marker_dir+'mti_'+R2_ph_dataset_date+'_marker_config.yaml')

positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
    base_marker_config_file=S1_marker_dir+'D500B_'+S1_ph_dataset_date+'_marker_config.yaml',tool_marker_config_file=S1_tcp_marker_dir+'positioner_tcp_marker_config.yaml')

#### change base H to calibrated ones ####
robot_scan_base = robot_weld.T_base_basemarker.inv()*robot_scan.T_base_basemarker
robot_scan.base_H = H_from_RT(robot_scan_base.R,robot_scan_base.p)
positioner_base = robot_weld.T_base_basemarker.inv()*positioner.T_base_basemarker
positioner.base_H = H_from_RT(positioner_base.R,positioner_base.p)
# T_to_base = Transform(np.eye(3),[0,0,-380])
# positioner.base_H = np.matmul(positioner.base_H,H_from_RT(T_to_base.R,T_to_base.p))
# input(positioner.base_H)

robot_weld.robot.P=deepcopy(robot_weld.calib_P)
robot_weld.robot.H=deepcopy(robot_weld.calib_H)
robot_scan.robot.P=deepcopy(robot_scan.calib_P)
robot_scan.robot.H=deepcopy(robot_scan.calib_H)
positioner.robot.P=deepcopy(positioner.calib_P)
positioner.robot.H=deepcopy(positioner.calib_H)

#### load R1 kinematic model
PH_data_dir='../mocap/PH_grad_data/test'+R1_ph_dataset_date+'_R1/train_data_'
with open(PH_data_dir+'calib_PH_q.pickle','rb') as file:
    PH_q=pickle.load(file)
nom_P=np.array([[0,0,0],[150,0,0],[0,0,760],\
                   [1082,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
nom_H=np.array([[0,0,1],[0,1,0],[0,-1,0],\
                [-1,0,0],[0,-1,0],[-1,0,0]]).T
ph_param_r1=PH_Param(nom_P,nom_H)
ph_param_r1.fit(PH_q,method='FBF')
ph_param_r1=None
#### load R2 kinematic model
PH_data_dir='../mocap/PH_grad_data/test'+R2_ph_dataset_date+'_R2/train_data_'
with open(PH_data_dir+'calib_PH_q.pickle','rb') as file:
    PH_q=pickle.load(file)
nom_P=np.array([[0,0,0],[155,0,0],[0,0,614],\
                   [640,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
nom_H=np.array([[0,0,1],[0,1,0],[0,-1,0],\
                [-1,0,0],[0,-1,0],[-1,0,0]]).T
ph_param_r2=PH_Param(nom_P,nom_H)
ph_param_r2.fit(PH_q,method='FBF')
ph_param_r2=None

#### data directory
# dataset='cup/'
# sliced_alg='circular_slice_shifted/'
# curve_data_dir = '../data/'+dataset+sliced_alg
# data_dir=curve_data_dir+'weld_scan_'+'2023_07_11_16_25_30'+'/'
# baselayer=False
# layer=367
# x=0

# dataset='blade0.1/'
# sliced_alg='auto_slice/'
# curve_data_dir = '../data/'+dataset+sliced_alg
# data_dir=curve_data_dir+'weld_scan_'+'2023_07_19_11_41_30'+'/'
# baselayer=True
# layer=1
# x=0

# dataset='blade0.1/'
# sliced_alg='auto_slice/'
# curve_data_dir = '../data/'+dataset+sliced_alg
# data_dir=curve_data_dir+'weld_scan_'+'2023_07_24_13_13_53'+'/'
# baselayer=False
# last_layer=365
# layer=384
# x=0

# dataset='blade0.1/'
# sliced_alg='auto_slice/'
# curve_data_dir = '../data/'+dataset+sliced_alg
# data_dir=curve_data_dir+'weld_scan_2023_09_05_18_32_02'+'/'
# baselayer=False
# last_layer=1
# layer=0
# x=0

dataset='circle_large/'
sliced_alg='static_stepwise_split/'
curve_data_dir = '../data/'+dataset+sliced_alg
data_dir=curve_data_dir+'weld_scan_2023_09_12_17_58_14'+'/'
baselayer=False
last_layer=153
layer=176
x=0

use_actual = False
regen_pcd = True

if not baselayer:
    layer_data_dir=data_dir+'layer_'+str(layer)+'_'+str(x)+'/'
    if layer!=0:
        last_layer_data_dir=data_dir+'layer_'+str(last_layer)+'_'+str(x)+'/'
    else:
        last_layer_data_dir=data_dir+'baselayer_'+str(last_layer)+'_'+str(x)+'/'
else:
    layer_data_dir=data_dir+'baselayer_'+str(layer)+'_'+str(x)+'/'
    last_layer_data_dir=data_dir+'baselayer_'+str(last_layer)+'_'+str(x)+'/'
out_scan_dir = layer_data_dir+'scans/'
last_out_scan_dir = last_layer_data_dir+'scans/'

if not baselayer:
    curve_sliced_relative=np.loadtxt(curve_data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
    curve_sliced_js=np.loadtxt(curve_data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
    positioner_js=np.loadtxt(curve_data_dir+'curve_sliced_js/D500B_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
else:
    curve_sliced_relative=np.loadtxt(curve_data_dir+'curve_sliced_relative/baselayer'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
    curve_sliced_js=np.loadtxt(curve_data_dir+'curve_sliced_js/MA2010_base_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
    positioner_js=np.loadtxt(curve_data_dir+'curve_sliced_js/D500B_base_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
rob_js_plan = np.hstack((curve_sliced_js,positioner_js))

### robot 1
weld_q_exe = np.loadtxt(layer_data_dir+'weld_js_exe.csv',delimiter=',')
if use_actual:
    curve_sliced_relative=[]
    for q_out in weld_q_exe[::-1]:
        Table_home_T = positioner.fwd(q_out[-2:])
        T_S1TCP_R1Base = np.matmul(positioner.base_H,H_from_RT(Table_home_T.R,Table_home_T.p))
        T_R1Base_S1TCP = np.linalg.inv(T_S1TCP_R1Base)
        ### R1 fwd
        opt_P,opt_H = ph_param_r1.predict(q_out[1:3])
        robot_weld.robot.P=opt_P
        robot_weld.robot.H=opt_H
        robot_T = robot_weld.fwd(q_out[:6])
        ###
        T_R1TCP_S1TCP = np.matmul(T_R1Base_S1TCP,H_from_RT(robot_T.R,robot_T.p))
        # print(robot_T)
        # input(T_R1TCP_S1TCP[:3,-1])
        if len(curve_sliced_relative)==0 or np.linalg.norm(T_R1TCP_S1TCP[:3,-1]-curve_sliced_relative[-1][:3])>0.8:
            if len(curve_sliced_relative)==0 or np.fabs(T_R1TCP_S1TCP[2,-1]-curve_sliced_relative[-1][2])<0.4:
                curve_sliced_relative.append(np.append(T_R1TCP_S1TCP[:3,-1],T_R1TCP_S1TCP[:3,2]))
    curve_sliced_relative=curve_sliced_relative[::-1]
    print("curve len:",len(curve_sliced_relative))
###########

scan_process = ScanProcess(robot_scan,positioner)
#### scanning process: processing point cloud and get h
if regen_pcd:
    with open(out_scan_dir+'mti_scans.pickle', 'rb') as file:
        mti_recording=pickle.load(file)
    q_out_exe=np.loadtxt(out_scan_dir+'scan_js_exe.csv',delimiter=',')
    robot_stamps=np.loadtxt(out_scan_dir+'scan_robot_stamps.csv',delimiter=',')
    crop_extend=15
    crop_min=tuple(np.min(curve_sliced_relative[:,:3],axis=0)-crop_extend)
    crop_max=tuple(np.max(curve_sliced_relative[:,:3],axis=0)+crop_extend)
    pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,use_calib=True,ph_param=ph_param_r2)
    # pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,use_calib=False)
    # visualize_pcd([pcd])
    pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
                                        min_bound=crop_min,max_bound=crop_max,outlier_remove=True,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=300)
else:
    pcd=o3d.io.read_point_cloud(out_scan_dir+'processed_pcd.pcd')
visualize_pcd([pcd])

# last pcd
if regen_pcd:
    with open(last_out_scan_dir+'mti_scans.pickle', 'rb') as file:
        mti_recording=pickle.load(file)
    q_out_exe=np.loadtxt(last_out_scan_dir+'scan_js_exe.csv',delimiter=',')
    robot_stamps=np.loadtxt(last_out_scan_dir+'scan_robot_stamps.csv',delimiter=',')
    #### scanning process: processing point cloud and get h
    curve_sliced_relative=np.array(curve_sliced_relative)
    crop_extend=15
    crop_min=tuple(np.min(curve_sliced_relative[:,:3],axis=0)-crop_extend)
    crop_max=tuple(np.max(curve_sliced_relative[:,:3],axis=0)+crop_extend)
    last_pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,use_calib=True,ph_param=ph_param_r2)
    # pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,use_calib=False)
    # visualize_pcd([last_pcd])
    last_pcd = scan_process.pcd_noise_remove(last_pcd,nb_neighbors=40,std_ratio=1.5,\
                                        min_bound=crop_min,max_bound=crop_max,outlier_remove=True,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=300)
else:
    last_pcd=o3d.io.read_point_cloud(last_out_scan_dir+'processed_pcd.pcd')
# visualize_pcd([last_pcd])

if use_actual:
    profile_height = scan_process.pcd2dh(pcd,last_pcd,curve_sliced_relative,drawing=True)
else:
    profile_height = scan_process.pcd2dh(pcd,last_pcd,curve_sliced_relative,robot_weld,rob_js_plan,ph_param=ph_param_r1,drawing=True)


curve_i=0
total_curve_i = len(profile_height)
for curve_i in range(total_curve_i):
    color_dist = plt.get_cmap("rainbow")(float(curve_i)/total_curve_i)
    plt.scatter(profile_height[curve_i,0],profile_height[curve_i,1],c=color_dist)
plt.xlabel('Lambda')
plt.ylabel('dh to Layer N (mm)')
plt.title("Height Profile")
plt.show()