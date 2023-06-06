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

class ScanProcess():
    def __init__(self,robot,positioner) -> None:
        self.robot=robot
        self.positioner=positioner
    
    def pcd_register(self,all_scan_points,scan_stamps,rob_js_exe,rob_stamps,voxel_size=0.1,static_positioner_q = np.radians([-60,180]),\
                     icp_threshold = 5,rmse_search_step=100):
        
        ## auto get timestamp where scans/robot start moving
        scan_start_rmse=0.7
        scan_move_stamp_i=None
        pcd_combined = None
        for scan_i in range(1,50):
            scan_points = deepcopy(all_scan_points[scan_i])
            pcd = o3d.geometry.PointCloud()
            pcd.points=o3d.utility.Vector3dVector(scan_points)
            if pcd_combined is None:
                pcd_combined=deepcopy(pcd)
            else:
                evaluation = o3d.pipelines.registration.evaluate_registration(
                            pcd_combined, pcd, 5, np.eye(4))
                if evaluation.inlier_rmse >= scan_start_rmse:
                    scan_move_stamp_i=scan_i
                    break
            pcd_combined=deepcopy(pcd)
        # release memory
        pcd_combined=None
        pcd=None
        scan_points=None
        ###
        rob_start_norm=1e-3*2
        rob_move_stamp_i=None
        for robt_i in range(10,50):
            if np.linalg.norm(np.degrees(rob_js_exe[robt_i]-rob_js_exe[robt_i-1]))>=rob_start_norm:
                rob_move_stamp_i=robt_i
                break
        
        print("Scan Start index:",scan_move_stamp_i) # where scans are different
        print("Robot Start index:",rob_move_stamp_i) # where robot starts moving

        self.robotner_t_diff=scan_stamps[scan_move_stamp_i]-rob_stamps[rob_move_stamp_i] ## (sec) timer start different between the robot and the scanner
        sca_stamps_sync_robt = scan_stamps-self.robotner_t_diff

        scan_points_t0 = deepcopy(all_scan_points[scan_move_stamp_i-1])
        #### t1 pcd
        robt_T = self.robot.fwd(rob_js_exe[rob_move_stamp_i-1][:6])
        scan_points_t0 = np.transpose(np.matmul(robt_T.R,np.transpose(scan_points_t0)))+robt_T.p
        ## pcd
        pcd_t0 = o3d.geometry.PointCloud()
        pcd_t0.points=o3d.utility.Vector3dVector(scan_points_t0)
        ## voxel down sample
        pcd_t0 = pcd_t0.voxel_down_sample(voxel_size=voxel_size)
        robt_move_t1_i=np.argsort(np.abs(rob_stamps-sca_stamps_sync_robt[scan_move_stamp_i]))[0]
        
        scan_points_t1_origin = deepcopy(all_scan_points[scan_move_stamp_i])
        rmse_i = 0
        rmse_low = 999
        all_rmse = []
        for search_i in range(0,rmse_search_step):
            # print(search_i)
            #### t1 pcd
            robt_T = self.robot.fwd(rob_js_exe[robt_move_t1_i+search_i][:6])
            scan_points_t1 = np.transpose(np.matmul(robt_T.R,np.transpose(scan_points_t1_origin)))+robt_T.p
            ## pcd
            pcd_t1 = o3d.geometry.PointCloud()
            pcd_t1.points=o3d.utility.Vector3dVector(scan_points_t1)
            ## voxel down sample
            pcd_t1 = pcd_t1.voxel_down_sample(voxel_size=voxel_size)

            evaluation = o3d.pipelines.registration.evaluate_registration(
                            pcd_t1, pcd_t0, icp_threshold, np.eye(4))
            if evaluation.inlier_rmse<rmse_low:
                rmse_low=evaluation.inlier_rmse
                rmse_i=search_i
            all_rmse.append(evaluation.inlier_rmse)
        ###
        # plt.plot(all_rmse,'o-')
        # plt.xlabel('Closest Timestamps Index')
        # plt.ylabel('Inlier RMSE')
        # plt.title('Inlier RMSE Using Robot Pose at Different Timestamps')
        # plt.show()
        # print(rmse_i)

        pcd_combined = None
        rob_js_exe_cor = []
        scan_i_start=None
        scan_N = len(scan_stamps) ## total scans
        for scan_i in range(scan_N):
            # discard scanner timestamp <0 (robot motion haven't start)
            if sca_stamps_sync_robt[scan_i]<0:
                rob_js_exe_cor.append(rob_js_exe[0])
                continue
            if scan_i_start is None:
                scan_i_start=scan_i

            scan_points = deepcopy(all_scan_points[scan_i])
            
            ## get corresponding js
            closest_i_sort=np.argsort(np.abs(rob_stamps-sca_stamps_sync_robt[scan_i]))
            closest_i = closest_i_sort[0]
            rob_js_exe_cor.append(rob_js_exe[closest_i])

            if len(rob_js_exe[closest_i])<=6:
                robt_T = self.robot.fwd(rob_js_exe[closest_i],world=True) # T_world^r2tool
                T_origin = self.positioner.fwd(static_positioner_q,world=True).inv() # T_tabletool^world
            else:
                robt_T = self.robot.fwd(rob_js_exe[closest_i][:6],world=True) # T_world^r2tool
                T_origin = self.positioner.fwd(rob_js_exe[closest_i][6:],world=True).inv() # T_tabletool^world
                # T_origin = self.positioner.fwd(static_positioner_q,world=True).inv()
            
            T_rob_positioner_top = T_origin*robt_T
            
            ## get the points closed to origin
            scan_points = np.transpose(np.matmul(T_rob_positioner_top.R,np.transpose(scan_points)))+T_rob_positioner_top.p
            # use legacy
            pcd = o3d.geometry.PointCloud()
            pcd.points=o3d.utility.Vector3dVector(scan_points)

            if pcd_combined is None:
                pcd_combined=deepcopy(pcd)
            else:
                pcd_combined+=pcd
        
        return pcd_combined
    
    def pcd_register_mti(self,all_scan_points,rob_js_exe,rob_stamps,voxel_size=0.05,static_positioner_q=np.radians([-60,180])):

        pcd_combined = None
        scan_N = len(rob_stamps) ## total scans
        for scan_i in range(scan_N):

            if len(rob_js_exe[scan_i])<=6:
                robt_T = self.robot.fwd(rob_js_exe[scan_i],world=True) # T_world^r2tool
                T_origin = self.positioner.fwd(static_positioner_q,world=True).inv() # T_tabletool^world
            else:
                robt_T = self.robot.fwd(rob_js_exe[scan_i][:6],world=True) # T_world^r2tool
                T_origin = self.positioner.fwd(rob_js_exe[scan_i][6:],world=True).inv() # T_tabletool^world
                # T_origin = turn_table.fwd(np.radians([-30,0]),world=True).inv()
            T_rob_positioner_top = T_origin*robt_T

            scan_points=deepcopy(all_scan_points[scan_i])
            scan_points = np.insert(scan_points,1,np.zeros(len(scan_points[0])),axis=0)
            scan_points = scan_points.T
            ## get the points closed to origin
            scan_points = np.transpose(np.matmul(T_rob_positioner_top.R,np.transpose(scan_points)))+T_rob_positioner_top.p
            # use legacy
            pcd = o3d.geometry.PointCloud()
            pcd.points=o3d.utility.Vector3dVector(scan_points)
            
            ## voxel down sample
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

            if pcd_combined is None:
                pcd_combined=deepcopy(pcd)
            else:
                pcd_combined+=pcd
        
        return pcd_combined
    
    def pcd_noise_remove(self,pcd_combined,voxel_down_flag=True,voxel_size=0.1,crop_flag=True,min_bound=(-50,-30,-10),max_bound=(50,30,50),\
                         outlier_remove=True,nb_neighbors=40,std_ratio=0.5,cluster_based_outlier_remove=True,cluster_neighbor=0.75,min_points=50*4):

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
        
        return pcd_combined

    def pcd2height(self,scanned_points,z_height_start,bbox_min=(-40,-20,0),bbox_max=(40,20,45),\
                   resolution_z=0.1,windows_z=0.2,resolution_x=0.1,windows_x=1,stop_thres=20,\
                   stop_thres_w=10,use_points_num=5,width_thres=0.8,Transz0_H=None):

        ##### cross section parameters
        # resolution_z=0.1
        # windows_z=0.2
        # resolution_x=0.1
        # windows_x=1
        # stop_thres=20
        # stop_thres_w=10
        # use_points_num=5 # use the largest/smallest N to compute w
        # width_thres=0.8 # prune width that is too close
        ###################################

        ###################### get the welding pieces ##################
        # This part will be replaced by welding path in the future
        ######## make the plane normal as z-axis
        if Transz0_H is None:
            ####### plane segmentation
            plane_model, inliers = scanned_points.segment_plane(distance_threshold=float(0.75),
                                                    ransac_n=int(5),
                                                    num_iterations=int(3000))
            ## Transform the plane to z=0
            plain_norm = plane_model[:3]/np.linalg.norm(plane_model[:3])
            k = np.cross(plain_norm,[0,0,1])
            k = k/np.linalg.norm(k)
            theta = np.arccos(plain_norm[2])
            Transz0 = Transform(rot(k,theta),[0,0,0])*\
                        Transform(np.eye(3),[0,0,plane_model[3]/plane_model[2]])
            Transz0_H=H_from_RT(Transz0.R,Transz0.p)
        scanned_points.transform(Transz0_H)
        ### now the distance to plane is the z axis

        # visualize_pcd([scanned_points])

        ## TODO:align path and scan
        # bbox_min=(-40,-20,0)
        # bbox_max=(40,20,45)
        ##################### get welding pieces end ########################

        ##### get projection of each z height
        profile_height = {}
        z_max=np.max(np.asarray(scanned_points.points)[:,2])
        for z in np.arange(z_height_start,z_max+resolution_z,resolution_z):
            #### crop z height
            min_bound = (-1e5,-1e5,z-windows_z/2)
            max_bound = (1e5,1e5,z+windows_z/2)
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
            points_proj=scanned_points.crop(bbox)
            ##################
            
            min_bound = bbox_min
            max_bound = bbox_max
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
            welds_points=points_proj.crop(bbox)

            # visualize_pcd([welds_points])

            #### get width with x-direction scanning
            if len(welds_points.points)<stop_thres:
                continue

            profile_p = []
            for x in np.arange(bbox_min[0],bbox_max[0]+resolution_x,resolution_x):
                min_bound = (x-windows_x/2,-1e5,-1e5)
                max_bound = (x+windows_x/2,1e5,1e5)
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
                welds_points_x = welds_points.crop(bbox)
                if len(welds_points_x.points)<stop_thres_w:
                    continue
                # visualize_pcd([welds_points_x])
                ### get the width
                sort_y=np.argsort(np.asarray(welds_points_x.points)[:,1])
                y_min_index=sort_y[:use_points_num]
                y_max_index=sort_y[-use_points_num:]
                y_mid_index=sort_y[use_points_num:-use_points_num]
                
                ### get y and prune y that is too closed
                y_min_all = np.asarray(welds_points_x.points)[y_min_index,1]
                y_min = np.mean(y_min_all)
                y_max_all = np.asarray(welds_points_x.points)[y_max_index,1]
                y_max = np.mean(y_max_all)

                actual_y_min_all=[]
                actual_y_max_all=[]
                for num_i in range(use_points_num):
                    if (y_max-y_min_all[num_i])>width_thres:
                        actual_y_min_all.append(y_min_all[num_i])
                    if (y_max_all[num_i]-y_min)>width_thres:
                        actual_y_max_all.append(y_max_all[num_i])
                #########
                y_max=0
                y_min=0
                if len(actual_y_max_all)!=0 and len(actual_y_min_all)!=0:
                    y_max=np.mean(actual_y_max_all)
                    y_min=np.mean(actual_y_min_all)

                this_width=y_max-y_min
                # z_height_ave = np.mean(np.asarray(welds_points_x.points)[np.append(y_min_index,y_max_index),2])
                z_height_ave = np.mean(np.asarray(welds_points_x.points)[:,2])
                profile_p.append(np.array([x,this_width,z_height_ave]))
            profile_p = np.array(profile_p)
            
            for pf_i in range(len(profile_p)):
                profile_height[profile_p[pf_i][0]] = profile_p[pf_i][2]

        profile_height_arr = []
        for x in profile_height.keys():
            profile_height_arr.append(np.array([x,profile_height[x]]))
        profile_height_arr=np.array(profile_height_arr)

        profile_height_arr_argsort = np.argsort(profile_height_arr[:,0])
        profile_height_arr=profile_height_arr[profile_height_arr_argsort]
        
        return profile_height_arr,Transz0_H

        