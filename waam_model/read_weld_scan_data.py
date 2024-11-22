from copy import deepcopy
from pathlib import Path
import pickle
import sys
sys.path.append('../scan/scan_tools/')
sys.path.append('../scan/scan_plan/')
sys.path.append('../scan/scan_process/')
from robotics_utils import *
from motoman_def import *
from scan_utils import *
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
    base_marker_config_file=config_dir+'MA2010_marker_config/MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config/weldgun_marker_config.yaml')
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA1440_marker_config/MA1440_marker_config.yaml')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'D500B_marker_config/D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config/positioner_tcp_marker_config.yaml')

Table_home_T = positioner.fwd(np.radians([-15,180]))
T_S1TCP_R1Base = np.linalg.inv(np.matmul(positioner.base_H,H_from_RT(Table_home_T.R,Table_home_T.p)))
T_R1Base_S1TCP = np.linalg.inv(T_S1TCP_R1Base)

#### change base H to calibrated ones ####
robot_scan.base_H = H_from_RT(robot_scan.T_base_basemarker.R,robot_scan.T_base_basemarker.p)
positioner.base_H = H_from_RT(positioner.T_base_basemarker.R,positioner.T_base_basemarker.p)
T_to_base = Transform(np.eye(3),[0,0,-380])
positioner.base_H = np.matmul(positioner.base_H,H_from_RT(T_to_base.R,T_to_base.p))

path_R=np.array([[-0.7071, 0.7071, -0.    ],
            [ 0.7071, 0.7071,  0.    ],
            [0.,      0.,     -1.    ]])
R_S1TCP = np.matmul(T_S1TCP_R1Base[:3,:3],path_R)

build_height_profile=False
plot_correction=False
plot_pcd = True
show_layer = []
# show_layer = [12]

x_lower = -99999
x_upper = 999999

# start_id=0
# end_id=-1

start_id=75
end_id=-75

# datasets=['baseline','correction','repeat 1','repeat 2']
datasets=['correction']
datasets_h_mean={}
datasets_h_std={}
for dataset in datasets:

    if dataset=='baseline':
        data_dir = '../data/wall_weld_test/moveL_100_baseline_weld_scan_2023_07_07_15_20_56/'
    elif dataset=='correction':
        # data_dir = '../data/wall_weld_test/moveL_160_noconstraints_weld_scan_2023_07_05_18_59_53/'
        data_dir = '../data/wall_weld_test/moveL_100_weld_scan_2023_07_24_11_19_58/'
        # data_dir = '../data/wall_weld_test/moveL_100_weld_scan_2023_08_02_15_17_25/'
    elif dataset=='repeat 1':
        data_dir = '../data/wall_weld_test/moveL_100_repeat_weld_scan_2023_08_02_16_03_50/'
    elif dataset=='repeat 2':
        data_dir = '../data/wall_weld_test/moveL_100_repeat_weld_scan_2023_08_02_17_07_02/'

    forward_flag=False
    all_profile_height=[]
    all_correction_layer=[]
    all_h_mean=[]
    all_h_std=[]
    pcd_wall = None
    for i in range(0,9999999):
        try:
            weld_dir=data_dir+'layer_'+str(i)+'/'
            weld_q=np.loadtxt(weld_dir+'weld_js_exe.csv',delimiter=',')
            weld_stamp=np.loadtxt(weld_dir+'weld_robot_stamps.csv',delimiter=',')
            scan_dir=weld_dir+'scans/'
            pcd = o3d.io.read_point_cloud(scan_dir+'processed_pcd.pcd')
            profile_height = np.load(scan_dir+'height_profile.npy')
            q_out_exe=np.loadtxt(scan_dir+'scan_js_exe.csv',delimiter=',')
            robot_stamps=np.loadtxt(scan_dir+'scan_robot_stamps.csv',delimiter=',')
            with open(scan_dir+'mti_scans.pickle', 'rb') as file:
                mti_recording=pickle.load(file)

            print("Layer",i)
            print("Forward:",not forward_flag)

            if pcd_wall is None:
                pcd_wall = deepcopy(pcd)
            else:
                pcd_wall = pcd_wall + pcd
                
        except:
            break


        visualize_pcd([pcd])
        pcd.farthest_point_sampling(100000)


        all_h_mean.append(np.mean(profile_height[start_id:end_id,1]))
        # all_h_mean.append(np.mean(profile_height[75:-75,1]))
        # print(len(profile_height[75:-75,1]))

        all_h_std.append(np.std(profile_height[start_id:end_id,1]))

    i=0
    m_size=12
    # print('all_correction_layer',all_correction_layer)
    # print('all_profile_height',all_profile_height)
    for profile_height in all_profile_height:
        if i in all_correction_layer:
            if i==all_correction_layer[0]:
                plt.scatter(profile_height[start_id:end_id,0],profile_height[start_id:end_id,1],s=3,c='tab:green',label='Corrected Layer')
            else:
                plt.scatter(profile_height[start_id:end_id,0],profile_height[start_id:end_id,1],s=3,c='tab:green')
        else:
            if i==0:
                plt.scatter(profile_height[start_id:end_id,0],profile_height[start_id:end_id,1],s=3,c='tab:blue',label='Forward (Right to Left)')
            elif i==1:
                plt.scatter(profile_height[start_id:end_id,0],profile_height[start_id:end_id,1],s=3,c='tab:orange',label='Backward (Left to Right)')
            elif i%2==0:
                plt.scatter(profile_height[start_id:end_id,0],profile_height[start_id:end_id,1],s=3,c='tab:blue')
            else:
                plt.scatter(profile_height[start_id:end_id,0],profile_height[start_id:end_id,1],s=3,c='tab:orange')
        i+=1
    plt.xlabel('x-axis')
    plt.ylabel('z-axis')
    plt.legend()
    plt.title("Height Profile")
    plt.tight_layout()
    plt.show()

    # keep the pcd_wall with points within +- y=10
    # pcd_wall_arr = np.asarray(pcd_wall.points)
    # pcd_wall_arr = pcd_wall_arr[np.where(pcd_wall_arr[:,1]>-10)[0]]
    # pcd_wall_arr = pcd_wall_arr[np.where(pcd_wall_arr[:,1]<10)[0]]
    # pcd_wall = o3d.geometry.PointCloud()
    # pcd_wall.points = o3d.utility.Vector3dVector(pcd_wall_arr)
    # # visualize the wall
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     pcd_wall.estimate_normals()
    #     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    #         pcd_wall, depth=9)
    #     mesh.compute_vertex_normals()
    visualize_pcd([pcd_wall])
    # save pcd to data directory
    o3d.io.write_point_cloud(data_dir+'pcd_wall.pcd', pcd_wall)