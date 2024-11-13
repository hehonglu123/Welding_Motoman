from copy import deepcopy
from pathlib import Path
import pickle
import sys
sys.path.append('../')
sys.path.append('../../toolbox/')
sys.path.append('../../scan/scan_tools/')
sys.path.append('../../scan/scan_plan/')
sys.path.append('../../scan/scan_process/')
from robot_def import *
from scan_utils import *
from scan_continuous import *
# from scanPathGen import *
from scanProcess import *

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import yaml
import open3d as o3d

zero_config=np.zeros(6)
# 0. robots. Note use "(robot)_pose_mocapcalib.csv"
config_dir='../../config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'torch.csv',\
    pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA2010_marker_config/MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config/weldgun_marker_config.yaml')
robot_scan=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'fujicam.csv',\
    pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
robot_flir=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
	pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'D500B_marker_config/D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config/positioner_tcp_marker_config.yaml')

build_pcd = False
get_height_profile = True
viz_height_profile = True
viz_pcd = True

datasets = ['torch_ori0_2024_09_25_14_13_31','torch_ori-15_2024_09_25_13_54_15','torch_ori15_2024_09_25_14_03_51']
# datasets = ['torch_ori-15_2024_09_25_13_54_15']

# get cmap from matplotlib
cmap = plt.get_cmap('jet')

for dataset in datasets:
    print('Processing dataset: ',dataset)
    data_dir = '../../data/wall_weld_test/'+dataset+'/'
    # 1. read meta file
    meta_file=data_dir+'meta.yaml'
    with open(meta_file) as file:
        meta_info=yaml.load(file,Loader=yaml.FullLoader)

    scan_process = ScanProcess(robot_scan,positioner)
    all_height_profile = []
    all_pcd = []
    for layer_i in range(1,meta_info['total_weld_layer']):

        # get layer data dir
        layer_dir = data_dir + 'weldlayer_' + str(layer_i) + '/'

        if build_pcd:
            # get stamps, robot scan js and scan
            stamps_scan = np.loadtxt(layer_dir + 'timestamps_scan.csv', delimiter=',')
            robot_scan_js = np.loadtxt(layer_dir + 'robot_scan_js.csv', delimiter=',')
            positioner_js = np.loadtxt(layer_dir + 'positioner_js.csv', delimiter=',')
            with open(layer_dir + 'scan.pkl', 'rb') as f:
                scan = pickle.load(f)
            
            # for scan_step in scan[len(scan)//2:]:
            #     plt.scatter(scan_step[:,0],scan_step[:,1])
            #     plt.show()
            # exit()

            z_height_start = meta_info['nominal_weld_height']*layer_i + meta_info['total_base_layer']*meta_info['nominal_base_height']
            crop_min=(-42.5-10,-30,-10)
            crop_max=(42.5+10,30,z_height_start+10)

            robot_scan_js = np.hstack((robot_scan_js, positioner_js))

            pcd = scan_process.pcd_register_mti(scan,robot_scan_js,stamps_scan,flip=True,scanner='fujicam')
            # open3d point cloud transform +y 40 mm
            pcd = pcd.translate((0,40,0))
            pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
                                                        min_bound=crop_min,max_bound=crop_max,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=100)
            # save pcd
            o3d.io.write_point_cloud(layer_dir+'pcd.pcd',pcd)
        else:
            pcd = o3d.io.read_point_cloud(layer_dir+'pcd.pcd')

        pcd = pcd.paint_uniform_color(cmap(layer_i/(meta_info['total_weld_layer']-1))[:3])
        all_pcd.append(pcd)

        # visualize_pcd([pcd])

        if get_height_profile:
            profile_height,_ = scan_process.pcd2height(deepcopy(pcd),-1,windows_x=0.4)
            np.savetxt(layer_dir+'profile_height.csv',profile_height,delimiter=',')
        else:
            profile_height = np.loadtxt(layer_dir+'profile_height.csv',delimiter=',')
        all_height_profile.append(profile_height)
            
    visualize_pcd([pcd])
    if viz_pcd:
        visualize_pcd(all_pcd)
    if viz_height_profile:
        for profile_height in all_height_profile:
            plt.plot(profile_height[250:551,0],profile_height[250:551,1])
        plt.show()

    dh_layers = []
    for layer_i in range(1,meta_info['total_weld_layer']):
        dh_layers.append(all_height_profile[layer_i][250:551,1]-all_height_profile[layer_i-1][250:551,1])
    
    print('dh mean:',np.mean(dh_layers))
    print('dh std:',np.std(dh_layers))