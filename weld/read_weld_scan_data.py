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
from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import open3d as o3d

def robot_weld_path_gen(all_layer_z,forward_flag,base_layer):
    R=np.array([[-0.7071, 0.7071, -0.    ],
            [ 0.7071, 0.7071,  0.    ],
            [0.,      0.,     -1.    ]])
    x0 =  1684	# Origin x coordinate
    y0 = -1179 + 428	# Origin y coordinate
    z0 = -260   # 10 mm distance to base

    weld_p=[]
    if base_layer: # base layer
        weld_p.append([x0 - 33, y0 - 20, z0+10])
        weld_p.append([x0 - 33, y0 - 20, z0])
        weld_p.append([x0 - 33, y0 - 105 , z0])
        weld_p.append([x0 - 33, y0 - 105 , z0+10])
    else: # top layer
        weld_p.append([x0 - 33, y0 - 30, z0+10])
        weld_p.append([x0 - 33, y0 - 30, z0])
        weld_p.append([x0 - 33, y0 - 95 , z0])
        weld_p.append([x0 - 33, y0 - 95 , z0+10])

    if not forward_flag:
        weld_p = weld_p[::-1]

    all_path_T=[]
    for layer_z in all_layer_z:
        path_T=[]
        for p in weld_p:
            path_T.append(Transform(R,p+np.array([0,0,layer_z])))

        all_path_T.append(path_T)
    
    return all_path_T

zero_config=np.zeros(6)
# 0. robots. Note use "(robot)_pose_mocapcalib.csv"
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

data_dir = '../data/wall_weld_test/full_test_weld_scan_2023_05_31_16_12_02/'
# data_dir='../data/wall_weld_test/top_layer_test_mti/scans/'

show_layer = [9,12]
forward_flag=False

all_profile_height=[]

for i in range(9,30):
    try:
        pcd = o3d.io.read_point_cloud(data_dir+'layer_'+str(i)+'/scans/processed_pcd.pcd')
        profile_height = np.load(data_dir+'layer_'+str(i)+'/scans/height_profile.npy')
        q_out_exe=np.loadtxt(data_dir +'layer_'+str(i)+ '/scans/scan_js_exe.csv',delimiter=',')
        robot_stamps=np.loadtxt(data_dir +'layer_'+str(i)+ '/scans/robot_stamps.csv',delimiter=',')
        with open(data_dir +'layer_'+str(i)+ '/scans/mti_scans.pickle', 'rb') as file:
            mti_recording=pickle.load(file)
        
        # q_out_exe=np.loadtxt(data_dir +'scan_js_exe.csv',delimiter=',')
        # robot_stamps=np.loadtxt(data_dir +'robot_stamps.csv',delimiter=',')
        # with open(data_dir +'mti_scans.pickle', 'rb') as file:
        #     mti_recording=pickle.load(file)
    except:
        break
    print("Layer",i)
    print("Forward:",not forward_flag)

    curve_x_start=43
    curve_x_end=-41
    crop_extend=10
    z_height_start=20
    scan_process = ScanProcess(robot_scan,positioner)
    crop_min=(curve_x_end-crop_extend,-30,-10)
    crop_max=(curve_x_start+crop_extend,30,z_height_start+30)
    crop_h_min=(curve_x_end-crop_extend,-20,-10)
    crop_h_max=(curve_x_start+crop_extend,20,z_height_start+30)
    q_init_table=np.radians([-15,200])
    pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,static_positioner_q=q_init_table)
    # visualize_pcd([pcd])
    pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
                                        min_bound=crop_min,max_bound=crop_max,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=100)
    # visualize_pcd([pcd])
    profile_height = scan_process.pcd2height(deepcopy(pcd),-1)

    all_profile_height.append(profile_height)

    profile_slope = np.gradient(profile_height[:,1])/np.gradient(profile_height[:,0])
    # profile_slope = np.append(profile_slope[0],profile_slope)
    # find slope peak
    peak_threshold=0.6
    weld_terrain=[]
    last_peak_i=None
    lastlast_peak_i=None
    for sample_i in range(len(profile_slope)):
        if np.fabs(profile_slope[sample_i])<peak_threshold:
            weld_terrain.append(0)
        else:
            if profile_slope[sample_i]>=peak_threshold:
                weld_terrain.append(1)
            elif profile_slope[sample_i]<=-peak_threshold:
                weld_terrain.append(-1)
            if lastlast_peak_i:
                
                if (weld_terrain[-1]==weld_terrain[lastlast_peak_i]) and (weld_terrain[-1]!=weld_terrain[last_peak_i]):
                    weld_terrain[last_peak_i]=0
            lastlast_peak_i=last_peak_i
            last_peak_i=sample_i

    weld_terrain=np.array(weld_terrain)
    weld_peak=[]
    last_peak=None
    last_peak_i=None
    flat_threshold=2.5
    for sample_i in range(len(profile_slope)):
        if weld_terrain[sample_i]!=0:
            if last_peak is None:
                weld_peak.append(profile_height[sample_i])
            else:
                # if the terrain change
                if (last_peak>0 and weld_terrain[sample_i]<0) or (last_peak<0 and weld_terrain[sample_i]>0):
                    weld_peak.append(profile_height[last_peak_i])
                    weld_peak.append(profile_height[sample_i])
                else:
                    # the terrain not change but flat too long
                    if profile_height[sample_i,0]-profile_height[last_peak_i,0]>flat_threshold:
                        weld_peak.append(profile_height[last_peak_i])
                        weld_peak.append(profile_height[sample_i])
            last_peak=deepcopy(weld_terrain[sample_i])
            last_peak_i=sample_i
    weld_peak=np.array(weld_peak)

    if not forward_flag:
        weld_bp = weld_peak[np.arange(0,len(weld_peak)-1,2)+1]
    else:
        weld_bp = weld_peak[np.arange(0,len(weld_peak),2)][::-1]

    if i in show_layer:
        # plot pcd
        visualize_pcd([pcd])
        # plot height profile
        plt.scatter(profile_height[:,0],profile_height[:,1])
        plt.xlabel('x-axis')
        plt.ylabel('z-axis')
        plt.title("Height Profile of Layer "+str(i))
        plt.show()

        plt.scatter(profile_height[:,0],profile_height[:,1]-np.mean(profile_height[:,1]),label='Height Profile')
        plt.plot(profile_height[:,0],profile_slope,label='Slope')
        plt.scatter(weld_peak[:,0],weld_peak[:,1]-np.mean(profile_height[:,1]))
        plt.scatter(weld_bp[:,0],weld_bp[:,1]-np.mean(profile_height[:,1]),label='Welding Speed Change Point')
        plt.xlabel('x-axis')
        plt.ylabel('z-axis')
        plt.legend()
        plt.show()
    
    forward_flag= not forward_flag

for profile_height in all_profile_height:
    plt.plot(profile_height[:,0],profile_height[:,1])
plt.xlabel('x-axis')
plt.ylabel('z-axis')
plt.title("Height Profile")
plt.show()