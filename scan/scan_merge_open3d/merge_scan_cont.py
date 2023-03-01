import sys
import matplotlib
sys.path.append('../../toolbox/')
sys.path.append('../scan_tools/')
from robot_def import *
from scan_utils import *
from general_robotics_toolbox import *
import open3d as o3d

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
from copy import deepcopy
import colorsys
import math

### open3d device ###
if o3d.core.cuda.is_available():
    device = o3d.core.Device("CUDA:0")
else:
    device = o3d.core.Device("CPU:0")
dtype = o3d.core.float32
#####################

data_dir='../../data/wall_weld_test/scan_cont_3/'
config_dir='../../config/'

robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg.csv')
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'scanner_tcp2.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg.csv')
turn_table=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',base_transformation_file=config_dir+'D500B_pose.csv',\
    pulse2deg_file_path=config_dir+'D500B_pulse2deg.csv')

scan_js_exe = np.loadtxt(data_dir+'scan_js_exe.csv',delimiter=",", dtype=np.float64)
rob_stamps = np.loadtxt(data_dir+'robot_stamps.csv',delimiter=",", dtype=np.float64)
sca_stamps = np.loadtxt(data_dir+'scan_stamps.csv',delimiter=",", dtype=np.float64)

scan_N = len(sca_stamps) ## total scans

## TODO:auto get timestamp where scans/robot start moving
# pcd_combined = o3d.t.geometry.PointCloud(device)
# for scan_i in range(scan_N):
#     scan_points = np.load(data_dir + 'points_'+str(scan_i)+'.npy')

#     pcd = o3d.t.geometry.PointCloud(device)
#     pcd.point.positions=o3d.core.Tensor(scan_points, dtype, device)
#     pcd=pcd.paint_uniform_color([0.8, 0.0, 0.0])

#     if scan_i==0:
#         pcd_combined=pcd.clone()
#     else:
#         pcd_combined=pcd_combined.paint_uniform_color([0, 0.8, 0.0])
#         pcd_combined=pcd_combined.append(pcd)
# for robt_i in range(1,1000):
#     print(np.degrees(scan_js_exe[robt_i]-scan_js_exe[robt_i-1]))
# exit()

scan_move_stamp_i=18 # where scans are different
rob_move_stamp_i=11 # where robot starts moving
robot_scanner_t_diff=sca_stamps[scan_move_stamp_i]-rob_stamps[rob_move_stamp_i] ## (sec) timer start different between the robot and the scanner
delay_scanner_loop=0.04 ## (sec) timer delay record in scanner loop (see scan_continuous.py)
sca_stamps_sync_robt = sca_stamps-delay_scanner_loop-robot_scanner_t_diff

T_origin_R=rot([0,0,1],np.radians(87.5))
T_origin=Transform(T_origin_R,np.array([200,-800,300]))

### using tensor or legacy ###
use_tensor=False
##############################

### process parameters
use_icp=True
voxel_size=0.1
######################

pcd_combined = None
scan_js_exe_cor = []
# scan_N=50
for scan_i in range(scan_N):
    # discard scanner timestamp <0 (robot motion haven't start)
    if sca_stamps_sync_robt[scan_i]<0:
        scan_js_exe_cor.append(scan_js_exe[0])
        continue
    scan_points = np.load(data_dir + 'points_'+str(scan_i)+'.npy')
    ## get corresponding js
    closest_i = np.argsort(np.abs(rob_stamps-sca_stamps_sync_robt[scan_i]))[0]
    scan_js_exe_cor.append(scan_js_exe[closest_i])
    robt_T = robot_scan.fwd(scan_js_exe[closest_i])
    scan_points = np.transpose(np.matmul(robt_T.R,np.transpose(scan_points)))+robt_T.p
    ## get the points closed to origin
    scan_points = np.transpose(np.matmul(T_origin.R,np.transpose(scan_points)))+T_origin.p

    if use_tensor:
        # use tensor
        pcd = o3d.t.geometry.PointCloud(device)
        pcd.point.positions=o3d.core.Tensor(scan_points, dtype, device)
    else:
        # use legacy
        pcd = o3d.geometry.PointCloud()
        pcd.points=o3d.utility.Vector3dVector(scan_points)
    
    ## voxel down sample
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    if pcd_combined is None:
        if use_tensor:
            pcd_combined=pcd.clone()
        else:
            pcd_combined=deepcopy(pcd)
    else:
        if use_icp:
            ###ICP
            if scan_i>23:
                print(scan_i)
                trans_init=np.eye(4)
                threshold = 5
                if use_tensor:
                    reg_p2p = o3d.t.pipelines.registration.icp(
                    pcd, pcd_combined, threshold, trans_init,
                    o3d.t.pipelines.registration.TransformationEstimationPointToPoint())
                else:
                    reg_p2p = o3d.pipelines.registration.registration_icp(
                    pcd, pcd_combined, threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint())
                pcd=pcd.transform(reg_p2p.transformation)
        ####################################
        if use_tensor:
            pcd_combined=pcd_combined.append(pcd)
        else:
            pcd_combined+=pcd

pcd_combined = pcd_combined.voxel_down_sample(voxel_size=0.5)
visualize_pcd([pcd_combined])






