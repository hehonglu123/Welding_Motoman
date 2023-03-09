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

data_dir='../../data/wall_weld_test/scan_cont_6/scans/'
config_dir='../../config/'

robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',\
    pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg.csv')
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'scanner_tcp2.csv',\
    base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg.csv')
turn_table=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',base_transformation_file=config_dir+'D500B_pose.csv',\
    pulse2deg_file_path=config_dir+'D500B_pulse2deg.csv')

scan_js_exe = np.loadtxt(data_dir+'scan_js_exe.csv',delimiter=",", dtype=np.float64)
rob_stamps = np.loadtxt(data_dir+'robot_stamps.csv',delimiter=",", dtype=np.float64)
rob_stamps=rob_stamps-rob_stamps[0]
sca_stamps = np.loadtxt(data_dir+'scan_stamps.csv',delimiter=",", dtype=np.float64)
sca_stamps=sca_stamps-sca_stamps[0]

scan_N = len(sca_stamps) ## total scans

## TODO:auto get timestamp where scans/robot start moving
# pcd_combined = None
# for scan_i in range(15,scan_N):
#     print(scan_i)
#     scan_points = np.load(data_dir + 'points_'+str(scan_i)+'.npy')

#     pcd = o3d.t.geometry.PointCloud(device)
#     pcd.point.positions=o3d.core.Tensor(scan_points, dtype, device)
#     pcd=pcd.paint_uniform_color([0.8, 0.0, 0.0])

#     if pcd_combined is None:
#         pcd_combined=pcd.clone()
#     else:
#         pcd_combined=pcd_combined.paint_uniform_color([0, 0.8, 0.0])
#         pcd_combined=pcd_combined.append(pcd)
    
#     visualize_pcd([pcd_combined])

# for robt_i in range(1,50):
#     print("robt i:",robt_i)
#     print(rob_stamps[robt_i])
#     print(np.degrees(scan_js_exe[robt_i]-scan_js_exe[robt_i-1]))
# exit()

scan_move_stamp_i=22 # where scans are different

# sca_stamps=np.append(sca_stamps[0],sca_stamps)
# scan_move_stamp_i=20 # where scans are different

rob_move_stamp_i=8 # where robot starts moving
robot_scanner_t_diff=sca_stamps[scan_move_stamp_i]-rob_stamps[rob_move_stamp_i] ## (sec) timer start different between the robot and the scanner
delay_scanner_loop=0 ## (sec) timer delay record in scanner loop (see scan_continuous.py)
sca_stamps_sync_robt = sca_stamps-delay_scanner_loop-robot_scanner_t_diff
# sca_stamps_sync_robt = sca_stamps-delay_scanner_loop-robot_scanner_t_diff-0.10863497946777212

# print(rob_stamps[:30])
# print(sca_stamps_sync_robt[:30])
# exit()

T_origin_R=rot([0,0,1],np.radians(87.5))
T_origin=Transform(T_origin_R,np.array([200,-800,300]))
T_origin_inv=T_origin.inv()

### using tensor or legacy ###
use_tensor=False
##############################

### process parameters
use_icp=False
timestep_search=False
timestep_search_2=False
search_start_i=50

threshold = 5
rmse_search_step=20
voxel_size=0.1
######################

####
scan_points_t0_origin = np.load(data_dir + 'points_'+str(scan_move_stamp_i-1)+'.npy')
#### t1 pcd
robt_T = robot_scan.fwd(scan_js_exe[rob_move_stamp_i-1])
scan_points_t0 = np.transpose(np.matmul(robt_T.R,np.transpose(scan_points_t0_origin)))+robt_T.p
## get the points closed to origin
scan_points_t0 = np.transpose(np.matmul(T_origin.R,np.transpose(scan_points_t0)))+T_origin.p
## pcd
pcd_t0 = o3d.geometry.PointCloud()
pcd_t0.points=o3d.utility.Vector3dVector(scan_points_t0)
## voxel down sample
pcd_t0 = pcd_t0.voxel_down_sample(voxel_size=voxel_size)
robt_move_t1_i=np.argsort(np.abs(rob_stamps-sca_stamps_sync_robt[scan_move_stamp_i]))[0]
# print(robt_move_t1_i)
scan_points_t1_origin = np.load(data_dir + 'points_'+str(scan_move_stamp_i)+'.npy')

rmse_i = 0
rmse_low = 999
for search_i in range(0,rmse_search_step):
    # print(search_i)
    #### t1 pcd
    robt_T = robot_scan.fwd(scan_js_exe[robt_move_t1_i+search_i])
    scan_points_t1 = np.transpose(np.matmul(robt_T.R,np.transpose(scan_points_t1_origin)))+robt_T.p
    ## get the points closed to origin
    scan_points_t1 = np.transpose(np.matmul(T_origin.R,np.transpose(scan_points_t1)))+T_origin.p
    ## pcd
    pcd_t1 = o3d.geometry.PointCloud()
    pcd_t1.points=o3d.utility.Vector3dVector(scan_points_t1)
    ## voxel down sample
    pcd_t1 = pcd_t1.voxel_down_sample(voxel_size=voxel_size)

    evaluation = o3d.pipelines.registration.evaluate_registration(
                    pcd_t1, pcd_t0, threshold, np.eye(4))
    if evaluation.inlier_rmse<rmse_low:
        rmse_low=evaluation.inlier_rmse
        rmse_i=search_i

    # print(evaluation)
    pcd_t0.paint_uniform_color([0,1,0])
    pcd_t1.paint_uniform_color([1,0,0])
    # if evaluation.inlier_rmse<0.787:
    # visualize_pcd([pcd_t1,pcd_t0])
###
# exit()
# rmse_i+=2
sca_stamps_sync_robt=sca_stamps_sync_robt+(rob_stamps[robt_move_t1_i+rmse_i]-sca_stamps_sync_robt[scan_move_stamp_i])
# robt_move_t1_i=np.argsort(np.abs(rob_stamps-sca_stamps_sync_robt[scan_move_stamp_i]))[0]
# print(robt_move_t1_i)
# exit()


pcd_combined = None
scan_js_exe_cor = []
scan_i_start=None
# scan_N=100
for scan_i in range(scan_N):
    print("Scan:",scan_i)
    # discard scanner timestamp <0 (robot motion haven't start)
    if sca_stamps_sync_robt[scan_i]<0:
        scan_js_exe_cor.append(scan_js_exe[0])
        continue
    if scan_i_start is None:
        scan_i_start=scan_i
    scan_points_origin = np.load(data_dir + 'points_'+str(scan_i)+'.npy')
    ## get corresponding js
    closest_i_sort=np.argsort(np.abs(rob_stamps-sca_stamps_sync_robt[scan_i]))
    closest_i = closest_i_sort[0]
    scan_js_exe_cor.append(scan_js_exe[closest_i])
    robt_T = robot_scan.fwd(scan_js_exe[closest_i])
    scan_points = np.transpose(np.matmul(robt_T.R,np.transpose(scan_points_origin)))+robt_T.p
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

    ## paint pcd for visualization
    color_dist = plt.get_cmap("rainbow")((scan_i-scan_i_start)/(scan_N-scan_i_start))
    pcd = pcd.paint_uniform_color(color_dist[:3])

    if pcd_combined is None:
        if use_tensor:
            pcd_combined=pcd.clone()
        else:
            pcd_combined=deepcopy(pcd)
    else:
        if scan_i>search_start_i:
            if use_icp:
                ###ICP
                if use_tensor:
                    reg_p2p = o3d.t.pipelines.registration.icp(
                    pcd, pcd_combined, threshold, np.eye(4),
                    o3d.t.pipelines.registration.TransformationEstimationPointToPoint())
                else:
                    reg_p2p = o3d.pipelines.registration.registration_icp(
                    pcd, pcd_combined, threshold, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint())
                pcd=pcd.transform(reg_p2p.transformation)
            
            if timestep_search:
                rmse_lowest = 999
                rmse_lowest_i = 0
                for iter in range(rmse_search_step):
                    color_dist = plt.get_cmap("gnuplot")(iter/rmse_search_step)
                    # print('Scan:',scan_i,",Iter:",iter,"")
                    # pcd_eva = pcd_combined.voxel_down_sample(voxel_size=0.5)
                    evaluation = o3d.pipelines.registration.evaluate_registration(
                        pcd, pcd_combined, threshold, np.eye(4))
                    # print(evaluation)
                    pcd = pcd.paint_uniform_color(color_dist[:3])

                    if evaluation.inlier_rmse<rmse_lowest:
                        rmse_lowest_i = iter
                        rmse_lowest=evaluation.inlier_rmse

                    if iter == rmse_search_step-1:
                        break
                    closest_i=closest_i_sort[iter+1]
                    robt_T = robot_scan.fwd(scan_js_exe[closest_i])
                    scan_points = np.transpose(np.matmul(robt_T.R,np.transpose(scan_points_origin)))+robt_T.p
                    ## get the points closed to origin
                    scan_points = np.transpose(np.matmul(T_origin.R,np.transpose(scan_points)))+T_origin.p
                    pcd.points=o3d.utility.Vector3dVector(scan_points)
                    ## voxel down sample
                    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

                closest_i=closest_i_sort[rmse_lowest_i]
                robt_T = robot_scan.fwd(scan_js_exe[closest_i])
                scan_points = np.transpose(np.matmul(robt_T.R,np.transpose(scan_points_origin)))+robt_T.p
                ## get the points closed to origin
                scan_points = np.transpose(np.matmul(T_origin.R,np.transpose(scan_points)))+T_origin.p
                pcd.points=o3d.utility.Vector3dVector(scan_points)
                ## voxel down sample
                pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
                print("Min RMSE:",rmse_lowest)
                print("Min RMSE i:",rmse_lowest_i)
                print("Scan stamps:",sca_stamps_sync_robt[scan_i])
                print("Robt stamps:",rob_stamps[closest_i])
                print("Robt-scan stamps:",rob_stamps[closest_i]-sca_stamps_sync_robt[scan_i])
                # pcd_last=pcd_last.paint_uniform_color([1,0,0])
                # pcd=pcd.paint_uniform_color([0,1,0])
                # pcd_cand=pcd_cand.paint_uniform_color([0,0,1])
                # visualize_pcd([pcd_last,pcd,pcd_cand])
                sca_stamps_sync_robt=sca_stamps_sync_robt+(rob_stamps[closest_i]-sca_stamps_sync_robt[scan_i])
                # timestep_search=False

            if timestep_search_2:
                rmse_lowest = 999
                rmse_lowest_i = 0
                pcd_cand= o3d.geometry.PointCloud()
                for iter in range(rmse_search_step):
                    evaluation = o3d.pipelines.registration.evaluate_registration(
                        pcd, pcd_last, threshold, np.eye(4))
                    # print(evaluation)

                    if evaluation.inlier_rmse<rmse_lowest:
                        rmse_lowest_i = iter
                        rmse_lowest=evaluation.inlier_rmse
                    
                    if iter == rmse_search_step-1:
                        break
                    closest_i=closest_i_sort[iter+1]
                    robt_T = robot_scan.fwd(scan_js_exe[closest_i])
                    scan_points = np.transpose(np.matmul(robt_T.R,np.transpose(scan_points_origin)))+robt_T.p
                    ## get the points closed to origin
                    scan_points = np.transpose(np.matmul(T_origin.R,np.transpose(scan_points)))+T_origin.p
                    pcd.points=o3d.utility.Vector3dVector(scan_points)
                    ## voxel down sample
                    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        ####################################
        if use_tensor:
            pcd_combined=pcd_combined.append(pcd)
        else:
            pcd_combined+=pcd
        # pcd_combined = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    pcd_last=deepcopy(pcd)

pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=0.5)
visualize_pcd([pcd_combined_down])

o3d.io.write_point_cloud(data_dir+'processed_pcd_raw.pcd',pcd_combined)

voxel_down_flag=True
crop_flag=False
outlier_remove=True
cluster_based_outlier_remove=True

####### processing parameters
voxel_size=0.1
## crop focused region
min_bound = (-1,-1,-1)
max_bound = (143.1+5,15.8+1,30.6+1)
## outlier removal
nb_neighbors=40
std_ratio=0.5
## clustering
cluster_neighbor=0.75
min_points=50
######################

#### processing
## voxel down sample
if voxel_down_flag:
    pcd_combined = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    print("Voxel Down done.")

## crop point clouds
if crop_flag:
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
    pcd_combined=pcd_combined.crop(bbox)

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




