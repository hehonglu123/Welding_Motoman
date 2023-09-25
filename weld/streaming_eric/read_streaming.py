import open3d as o3d

from copy import deepcopy
from pathlib import Path
import pickle
import sys
sys.path.append('../../toolbox/')
sys.path.append('../../scan/scan_tools/')
sys.path.append('../../scan/scan_plan/')
sys.path.append('../../scan/scan_process/')
sys.path.append('../../mocap/')
sys.path.append('../')
from robot_def import *
from scan_utils import *
from scan_continuous import *
from scanPathGen import *
from scanProcess import *
from PH_interp import *
from weldCorrectionStrategy import *
from matplotlib.animation import FuncAnimation
from functools import partial

from general_robotics_toolbox import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import glob
import yaml
from math import ceil,floor

from sklearn.cluster import DBSCAN

R1_ph_dataset_date='0801'
R2_ph_dataset_date='0801'
S1_ph_dataset_date='0801'

def get_slope(p1,p2):
    
    return (p2[1]-p1[1])/(p2[0]-p1[0])

zero_config=np.zeros(6)
# 0. robots.
config_dir='../../config/'
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

#### load R1 kinematic model
PH_data_dir='../../mocap/PH_grad_data/test'+R1_ph_dataset_date+'_R1/train_data_'
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
PH_data_dir='../../mocap/PH_grad_data/test'+R2_ph_dataset_date+'_R2/train_data_'
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
robot_weld.robot.P=deepcopy(robot_weld.calib_P)
robot_weld.robot.H=deepcopy(robot_weld.calib_H)
robot_scan.robot.P=deepcopy(robot_scan.calib_P)
robot_scan.robot.H=deepcopy(robot_scan.calib_H)
positioner.robot.P=deepcopy(positioner.calib_P)
positioner.robot.H=deepcopy(positioner.calib_H)

dataset='circle_large/'
sliced_alg='static_spiral/'
curve_data_dir = '../../data/'+dataset+sliced_alg
data_dir=curve_data_dir+'weld_scan_2023_09_20_21_11_14'+'/'

with open(data_dir+'mti_scans.pickle', 'rb') as file:
    mti_recording_all=pickle.load(file)
with open(data_dir+'robot_js.pickle', 'rb') as file:
    robot_js_all=pickle.load(file)

all_layers=[0,26,44,62,80,98,116,116]

regen_pcd=False
regen_dh=False
Transz0_H=None

chop_id = 50

total_layer=len(robot_js_all)
sec_num=0
all_pcd=o3d.geometry.PointCloud()
for layer_count in range(0,total_layer):
    layer=all_layers[0] if layer_count==0 else all_layers[layer_count-1] 
    curve_sliced_relative=np.loadtxt(curve_data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(sec_num)+'.csv',delimiter=',')
    
    robot_js=robot_js_all[layer_count]
    # wrap to -pi pi
    count=0
    ang=robot_js[0][-1]
    while ang<-np.pi:
        ang+=2*np.pi
        count+=1
    robot_js[:,-1]=robot_js[:,-1]+count*2*np.pi
    
    mti_recording=np.array(mti_recording_all[layer_count])
    
    this_mti_recording = deepcopy(mti_recording[robot_js[:,-1]>-np.pi])
    this_robot_js = deepcopy(robot_js[robot_js[:,-1]>-np.pi])
    if layer_count>0:
        layer_robot_js = np.vstack((last_robot_js,this_robot_js))
        layer_mti_recording=[]
        layer_mti_recording.extend(deepcopy(last_mti_recording))
        layer_mti_recording.extend(deepcopy(this_mti_recording))
    last_robot_js = deepcopy(robot_js[robot_js[:,-1]<=-np.pi])
    last_mti_recording = deepcopy(mti_recording[robot_js[:,-1]<=-np.pi])
    if layer_count>0:
        robot_js=layer_robot_js
        mti_recording=layer_mti_recording
    
    print(len(robot_js))
    print(len(mti_recording))
    
    # if layer_count!=0:
    #     mti_recording = mti_recording_all[layer_count].T[]
    
    robot_stamps=robot_js[:,0]
    print(robot_stamps[100:110])
    print(np.mean(np.diff(robot_stamps)))
    q_out_exe=robot_js[:,7:]
    # robot_stamps=
    
    if layer_count>1:
        dbscan = DBSCAN(eps=0.5,min_samples=20)
        fig = plt.figure()
        playback_speed=2
        def updatefig(i):
            print(i)
            fig.clear()
            
            ## remove not in interested region
            st=time.time()
            mti_pcd=np.delete(mti_recording[i*playback_speed],mti_recording[i*playback_speed][1]==0,axis=1)
            mti_pcd=np.delete(mti_pcd,mti_pcd[1]<85,axis=1)
            mti_pcd=np.delete(mti_pcd,mti_pcd[1]>100,axis=1)
            mti_pcd=np.delete(mti_pcd,mti_pcd[0]<-10,axis=1)
            mti_pcd=np.delete(mti_pcd,mti_pcd[0]>10,axis=1)
            mti_pcd = mti_pcd.T
            
            # mti_pcd_x_sort_id = np.argsort(mti_pcd[0])
            # mti_pcd_x = mti_pcd[0][mti_pcd_x_sort_id]
            # mti_pcd_z = mti_pcd[1][mti_pcd_x_sort_id]
            # mti_pcd = np.array([mti_pcd_x,mti_pcd_z]).T
            
            # cluster based noise remove
            dbscan.fit(mti_pcd)
            
            n_clusters_ = len(set(dbscan.labels_))
            
            print("Total T:",time.time()-st)
            
            ## pcd process
            # get slope
            # slope_thres = 
            # pcd_slope = np.divide(np.diff(mti_pcd_x),np.diff(mti_pcd_z))
            # pcd_slope = np.append(pcd_slope[0],pcd_slope)
            # pcd_slope_large = pcd_slope>
            
            # plt.scatter(-1*mti_pcd_x,mti_pcd_z)
            # plt.scatter(-1*mti_pcd_x,np.fabs(pcd_slope))
            for cluster_i in range(n_clusters_-1):
                cluster_id = dbscan.labels_==cluster_i
                plt.scatter(-1*mti_pcd[cluster_id][:,0],mti_pcd[cluster_id][:,1])
            plt.xlim((-30,30))
            plt.ylim((50,120))
            plt.draw()
        anim = FuncAnimation(fig, updatefig, np.floor(len(mti_recording)/playback_speed).astype(int),interval=30,repeat=False)
        plt.show()
    
    
    #### scanning process: processing point cloud and get h
    curve_sliced_relative=np.array(curve_sliced_relative)
    
    scan_process = ScanProcess(robot_scan,positioner)
    if regen_pcd:
        crop_extend=15
        crop_min=tuple(np.min(curve_sliced_relative[:,:3],axis=0)-crop_extend)
        crop_max=tuple(np.max(curve_sliced_relative[:,:3],axis=0)+crop_extend)
        pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,use_calib=True,ph_param=ph_param_r2)
        # pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,use_calib=False)
        visualize_pcd([pcd])
        pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
                                            min_bound=crop_min,max_bound=crop_max,outlier_remove=True,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=300)
        o3d.io.write_point_cloud(data_dir+'layer_'+str(layer_count)+'_pcd.pcd',pcd)
    else:
        pcd=o3d.io.read_point_cloud(data_dir+'layer_'+str(layer_count)+'_pcd.pcd')
    visualize_pcd([pcd])
    
    pcd,Transz0_H=scan_process.pcd_calib_z(pcd,Transz0_H=Transz0_H)
    if layer_count!=0:
        
        if regen_dh:
            profile_dh,profile_height = scan_process.pcd2dh_compare(pcd,last_pcd,curve_sliced_relative[::4],drawing=False)
            np.save(data_dir+'layer_'+str(layer_count)+'_dh_profile.npy',profile_dh)
            np.save(data_dir+'layer_'+str(layer_count)+'_height_profile.npy',profile_height)
        else:
            profile_dh=np.load(data_dir+'layer_'+str(layer_count)+'_dh_profile.npy')
            profile_height=np.load(data_dir+'layer_'+str(layer_count)+'_height_profile.npy')
            
        
        # profile_dh_first_half = 
        
        curve_i=0
        total_curve_i = len(profile_dh)
        ax = plt.figure().add_subplot()
        for curve_i in range(total_curve_i):
            color_dist = plt.get_cmap("rainbow")(float(curve_i)/total_curve_i)
            ax.scatter(profile_dh[curve_i,0],profile_dh[curve_i,1],c=color_dist)
        ax.set_xlabel('Lambda')
        ax.set_ylabel('dh to previous (mm)')
        ax.set_title("dH Profile")
        plt.show()
        
        curve_i=0
        total_curve_i = len(profile_height)
        ax = plt.figure().add_subplot()
        for curve_i in range(total_curve_i):
            color_dist = plt.get_cmap("rainbow")(float(curve_i)/total_curve_i)
            ax.scatter(profile_height[curve_i,0],profile_height[curve_i,1],c=color_dist)
        ax.set_xlabel('Lambda')
        ax.set_ylabel('dh to Layer N (mm)')
        ax.set_title("Height Profile")
        plt.show()
        
        print("Layer Count",layer_count)
        print("")
        
    
    all_pcd += pcd
    # visualize_pcd([all_pcd])
    last_pcd=pcd
    
if regen_pcd:
    o3d.io.write_point_cloud(data_dir+'full_pcd.pcd',all_pcd)

crop_min=tuple([-1e5,-1e5,-0.1])
crop_max=tuple([1e5,1e5,1e5])
bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=crop_min,max_bound=crop_max)
all_pcd=all_pcd.crop(bbox)
visualize_pcd([all_pcd])
