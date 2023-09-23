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
#### load S1 kinematic model

regen_pcd = False

#### data directory
dataset='circle_large/'
sliced_alg='static_stepwise_shift/'
curve_data_dir = '../data/'+dataset+sliced_alg
# data_dir=curve_data_dir+'weld_scan_baseline_2023_09_18_16_17_34'+'/'
data_dir=curve_data_dir+'weld_scan_correction_2023_09_19_16_32_31'+'/'

#### welding spec, goal
with open(curve_data_dir+'slicing.yml', 'r') as file:
    slicing_meta = yaml.safe_load(file)
line_resolution = slicing_meta['line_resolution']
total_layer = slicing_meta['num_layers']
total_baselayer = slicing_meta['num_baselayers']

all_layer_dir=glob.glob(data_dir+'layer_*_0')
total_print_layer = len(all_layer_dir)
total_count=total_print_layer

layer_num = []
for layer_count in range(0,total_count):
    # get printed layer number
    layer=all_layer_dir[layer_count].split('/')
    layer=layer[-1]
    layer=layer.split('\\')
    layer=layer[-1]
    layer=layer.split('_')
    layer=int(layer[1])
    layer_num.append(layer)
layer_num = np.sort(layer_num)

last_curve_relative = []
last_curve_height = []
all_pcd=o3d.geometry.PointCloud()
viz_obj=[]
Transz0_H=None

dh_std=[]
for layer_count in range(0,total_count):
    baselayer=False
    # if layer_count!= 0 and layer_count<=total_baselayer:
    #     baselayer=True
    
    # get printed layer number
    layer=int(layer_num[layer_count])
    
    print("Layer:",layer)
        
    layer_data_dir=data_dir+'layer_'+str(layer)+'_'
    
    num_sections = len(glob.glob(layer_data_dir+'*'))
    pcd_layer=o3d.geometry.PointCloud()
    
    layer_curve_relative=[]
    layer_curve_dh=[]
    for x in range(num_sections):
        layer_sec_data_dir=layer_data_dir+str(x)+'/'
        out_scan_dir = layer_sec_data_dir+'scans/'
        print(out_scan_dir)
        
        if layer<0:
            read_layer=0
        else:
            read_layer=layer
        if not baselayer:
            curve_sliced_relative=np.loadtxt(curve_data_dir+'curve_sliced_relative/slice'+str(read_layer)+'_'+str(x)+'.csv',delimiter=',')
            curve_sliced_js=np.loadtxt(curve_data_dir+'curve_sliced_js/MA2010_js'+str(read_layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
            positioner_js=np.loadtxt(curve_data_dir+'curve_sliced_js/D500B_js'+str(read_layer)+'_'+str(x)+'.csv',delimiter=',')
        else:
            curve_sliced_relative=np.loadtxt(curve_data_dir+'curve_sliced_relative/baselayer'+str(read_layer)+'_'+str(x)+'.csv',delimiter=',')
            curve_sliced_js=np.loadtxt(curve_data_dir+'curve_sliced_js/MA2010_base_js'+str(read_layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
            positioner_js=np.loadtxt(curve_data_dir+'curve_sliced_js/D500B_base_js'+str(read_layer)+'_'+str(x)+'.csv',delimiter=',')
        
        rob_js_plan = np.hstack((curve_sliced_js,positioner_js))
        
        with open(out_scan_dir+'mti_scans.pickle', 'rb') as file:
            mti_recording=pickle.load(file)
        q_out_exe=np.loadtxt(out_scan_dir+'scan_js_exe.csv',delimiter=',')
        robot_stamps=np.loadtxt(out_scan_dir+'scan_robot_stamps.csv',delimiter=',')

        scan_process = ScanProcess(robot_scan,positioner)
        if regen_pcd:
            #### scanning process: processing point cloud and get h
            crop_extend=15
            crop_min=tuple(np.min(curve_sliced_relative[:,:3],axis=0)-crop_extend)
            crop_max=tuple(np.max(curve_sliced_relative[:,:3],axis=0)+crop_extend)
            pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,use_calib=True,ph_param=ph_param_r2)
            # pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,use_calib=False)
            # visualize_pcd([pcd])
            cluser_minp = 300
            while True:
                pcd_new = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
                                                    min_bound=crop_min,max_bound=crop_max,outlier_remove=True,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=cluser_minp)
                visualize_pcd([pcd_new])
                while True:
                    q=input("Continue?")
                    if q=='':
                        break
                    try:
                        cluser_minp=int(q)
                        break
                    except:
                        continue
                if q=='':
                    break
            pcd=pcd_new
                    
        else:
            pcd=o3d.io.read_point_cloud(out_scan_dir+'processed_pcd.pcd')
        
        pcd,Transz0_H = scan_process.pcd_calib_z(pcd,Transz0_H=Transz0_H)
        
        # dh plot
        if layer!=-1:
            # profile_height = scan_process.pcd2dh(pcd,last_pcd,curve_sliced_relative,robot_weld,rob_js_plan,ph_param=ph_param_r1,drawing=True)
            profile_height = scan_process.pcd2dh(pcd,curve_sliced_relative,drawing=False)
            if len(layer_curve_dh)!=0:
                profile_height[:,0]+=layer_curve_dh[-1,0]
            layer_curve_dh.extend(profile_height)
        layer_curve_relative.extend(curve_sliced_relative)

        pcd_layer+=pcd
    
    # get the full layer height
    # if layer!=-1:
    #     layer_curve_height=scan_process.dh2height(layer_curve_relative,layer_curve_dh,last_curve_relative,last_curve_height)
    # else:
    #     layer_curve_height=np.zeros(len(layer_curve_relative))
    
    last_pcd=pcd_layer
    # last_curve_height=layer_curve_height
    # last_curve_relative=layer_curve_relative
    
    all_pcd=all_pcd+last_pcd
    
    layer_curve_relative=np.array(layer_curve_relative)
    
    # curve_p=[]
    # curve_R=[]
    # curve_i=0
    # for curve_wp in layer_curve_relative:
    #     if np.all(curve_wp==layer_curve_relative[-1]):
    #         wp_R = direction2R(-1*curve_wp[3:],curve_wp[:3]-layer_curve_relative[curve_i-1][:3])
    #     else:
    #         wp_R = direction2R(-1*curve_wp[3:],layer_curve_relative[curve_i+1][:3]-curve_wp[:3])
    #     curve_p.append(curve_wp[:3])
    #     curve_R.append(wp_R)
    #     curve_i+=1
    # path_viz_frames = visualize_frames(curve_R[:-20],curve_p[:-20],size=1,visualize=False,frame_obj=True)
    # viz_obj.extend(path_viz_frames)

    layer_curve_dh=np.array(layer_curve_dh)
    # curve_i=0
    # total_curve_i = len(layer_curve_dh)
    # ax = plt.figure().add_subplot()
    # for curve_i in range(total_curve_i):
    #     color_dist = plt.get_cmap("rainbow")(float(curve_i)/total_curve_i)
    #     ax.scatter(layer_curve_dh[curve_i,0],layer_curve_dh[curve_i,1],c=color_dist)
    # ax.set_xlabel('Lambda')
    # ax.set_ylabel('dh to Layer N (mm)')
    # ax.set_title("dH Profile")
    # plt.ion()
    # plt.show(block=False)
    
    # curve_i=0
    # total_curve_i = len(layer_curve_height)
    # layer_curve_relative=np.array(layer_curve_relative)
    # lam_curve = calc_lam_cs(layer_curve_relative[:,:3])
    # ax = plt.figure().add_subplot()
    # for curve_i in range(total_curve_i):
    #     color_dist = plt.get_cmap("rainbow")(float(curve_i)/total_curve_i)
    #     ax.scatter(lam_curve[curve_i],layer_curve_height[curve_i],c=color_dist)
    # ax.set_xlabel('Lambda')
    # ax.set_ylabel('Layer N Height (mm)')
    # ax.set_title("Height Profile")
    # plt.show(block=False)
    
    # input("Continue? ")
    # viz_list=deepcopy(viz_obj)
    # viz_list.append(all_pcd)
    # visualize_pcd(viz_list)
    
    if layer!=-1:
        dh_std.append(np.std(layer_curve_dh[:,1]))

# save std data
np.save(data_dir+'height_std.npy',dh_std)

# viz_obj.append(all_pcd)
# visualize_pcd(viz_obj)