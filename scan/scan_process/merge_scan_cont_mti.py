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
import pickle

def raw_data_filtering(single_scan, noise_filter=False):
    single_scan=single_scan.T
    ###delete noise centered at x=0
    single_scan=np.delete(single_scan,np.argwhere(abs(single_scan[:,0])==0).flatten(),axis=0)
    ###clip data within range
    indices=np.argwhere(single_scan[:,1]>50).flatten()
    single_scan=single_scan[indices]

    if noise_filter:
        ###filter out outlier noise
        outlier_indices=identify_outliers2(single_scan[:,1],rolling_window=20,threshold=1e-2)
        return np.delete(single_scan,outlier_indices,axis=0)
    else:
        single_scan=single_scan.T
        return single_scan

### open3d device ###
if o3d.core.cuda.is_available():
    device = o3d.core.Device("CUDA:0")
else:
    device = o3d.core.Device("CPU:0")
dtype = o3d.core.float32
#####################

# data_dir='../../data/wall_weld_test/scan_cont_6/scans/'
# data_dir='../../data/wall_weld_test/wall_param_1/scans/'
# data_dir='../../data/wall_weld_test/scan_cont_newdx_1/scans/'
# data_dir='../../data/wall_weld_test/wall_param_data_collection/path_Rz-45_Ry0_stand_off_d243_b_theta45_scan_angle-45_45_z0_35_/scans/'
data_dir='../../data/wall_weld_test/full_test_mti/scans/'
# data_dir='../../data/wall_weld_test/top_layer_test_mti/scans/'
config_dir='../../config/'

robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA1440_marker_config.yaml')
turn_table=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config.yaml')
#### change base H to calibrated ones ####
robot_scan.base_H = H_from_RT(robot_scan.T_base_basemarker.R,robot_scan.T_base_basemarker.p)
turn_table.base_H = H_from_RT(turn_table.T_base_basemarker.R,turn_table.T_base_basemarker.p)
T_to_base = Transform(np.eye(3),[0,0,-380])
turn_table.base_H = np.matmul(turn_table.base_H,H_from_RT(T_to_base.R,T_to_base.p))

# print(robot_scan.fwd(np.zeros(6)))
# exit()

scan_js_exe = np.loadtxt(data_dir+'scan_js_exe.csv',delimiter=",", dtype=np.float64)
rob_stamps = np.loadtxt(data_dir+'robot_stamps.csv',delimiter=",", dtype=np.float64)
rob_stamps=rob_stamps-rob_stamps[0]
with open(data_dir + 'mti_scans.pickle', 'rb') as file:
    mti_recording=pickle.load(file)

### process parameters
voxel_size=0.05
#####################

### read raw for checking
# raw_pcd = o3d.io.read_point_cloud(data_dir+'processed_pcd.pcd')
# visualize_pcd([raw_pcd])
# raw_pcd.
#########################

pcd_combined = None
scan_N = len(rob_stamps) ## total scans
# scan_N = int(len(rob_stamps)/3)
for scan_i in range(scan_N):
    print("Scan:",scan_i)

    if len(scan_js_exe[scan_i])<=6:
        robt_T = robot_scan.fwd(scan_js_exe[scan_i],world=True) # T_world^r2tool
        T_origin = turn_table.fwd(np.radians([-15,180]),world=True).inv() # T_tabletool^world
    else:
        robt_T = robot_scan.fwd(scan_js_exe[scan_i][:6],world=True) # T_world^r2tool
        T_origin = turn_table.fwd(scan_js_exe[scan_i][6:],world=True).inv() # T_tabletool^world
        # T_origin = turn_table.fwd(np.radians([-15,180]),world=True).inv()
    T_rob_positioner_top = T_origin*robt_T

    scan_points=deepcopy(mti_recording[scan_i])
    scan_points=raw_data_filtering(scan_points)
    # if scan_i==100:
    #     for i in range(len(scan_points[0])):
    #         print(scan_points[:,i])

    scan_points = np.insert(scan_points,1,np.zeros(len(scan_points[0])),axis=0)
    # scan_points[0] = -1*scan_points[0]
    scan_points = scan_points.T
    ## get the points closed to origin
    scan_points = np.transpose(np.matmul(T_rob_positioner_top.R,np.transpose(scan_points)))+T_rob_positioner_top.p
    # use legacy
    pcd = o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(scan_points)

    # if scan_i%100==0:
    #     visualize_pcd([pcd,raw_pcd])

    # if scan_i==100:
    #     print(np.degrees(scan_js_exe[scan_i]))
    #     visualize_pcd([pcd])
    #     exit()
    
    ## voxel down sample
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    if pcd_combined is None:
        pcd_combined=deepcopy(pcd)
    else:
        pcd_combined+=pcd
    # visualize_pcd([pcd_combined])

pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=0.5)
visualize_pcd([pcd_combined_down])
# exit()

o3d.io.write_point_cloud(data_dir+'processed_pcd_raw.pcd',pcd_combined)

voxel_down_flag=True
crop_flag=True
outlier_remove=True
cluster_based_outlier_remove=True

####### processing parameters
voxel_size=0.05
## crop focused region
min_bound = (-50,-30,-10)
max_bound = (50,30,50)
## outlier removal
nb_neighbors=10
std_ratio=0.85
## clustering
cluster_neighbor=1
min_points=100
######################

## crop point clouds
if crop_flag:
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
    pcd_combined=pcd_combined.crop(bbox)

#### processing
## voxel down sample
if voxel_down_flag:
    pcd_combined = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    print("Voxel Down done.")

if outlier_remove:
    cl,ind=pcd_combined.remove_statistical_outlier(nb_neighbors=nb_neighbors,std_ratio=std_ratio)
    # display_inlier_outlier(pcd_combined,ind)
    pcd_combined=cl
    print("Outlier Removal done.")

## DBSCAN pcd clustering
if cluster_based_outlier_remove:
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd_combined.cluster_dbscan(eps=cluster_neighbor, min_points=min_points, print_progress=True))
    max_label=labels.max()
    pcd_combined=pcd_combined.select_by_index(np.argwhere(labels>=0))
    print("Cluster based Outlier Removal done.")

# print("Generate pcd Total time:",time,time.perf_counter()-st+dt)

visualize_pcd([pcd_combined])
o3d.io.write_point_cloud(data_dir+'processed_pcd.pcd',pcd_combined)




