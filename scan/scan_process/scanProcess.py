import sys
import matplotlib
sys.path.append('../scan_tools/')
from motoman_def import *
from scan_utils import *
from robotics_utils import *
from lambda_calc import *
from general_robotics_toolbox import *
import open3d as o3d
from sklearn.cluster import DBSCAN
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
    
    def pcd_register_mti(self,all_scan_points,rob_js_exe,rob_stamps,voxel_size=0.05,static_positioner_q=np.radians([-60,180]),flip=False,scanner='mti',use_calib=False,ph_param=None):

        pcd_combined = None
        scan_N = len(rob_js_exe) ## total scans
        for scan_i in range(scan_N):

            if len(rob_js_exe[scan_i])<=6:
                if use_calib:
                    if ph_param is not None:
                        opt_P,opt_H = ph_param.predict(rob_js_exe[scan_i][1:3])
                        self.robot.robot.P=opt_P
                        self.robot.robot.H=opt_H
                    robt_T = self.robot.fwd(rob_js_exe[scan_i][:6],world=True) # T_world^r2tool
                else:
                    robt_T = self.robot.fwd(rob_js_exe[scan_i][:6],world=True) # T_world^r2tool
                T_origin = self.positioner.fwd(static_positioner_q,world=True).inv() # T_tabletool^world
            else:
                # print(np.degrees(rob_js_exe[scan_i][:6]))
                # print(np.degrees(self.robot.robot.joint_lower_limit))
                # print("===============")
                if use_calib:
                    if ph_param is not None:
                        opt_P,opt_H = ph_param.predict(rob_js_exe[scan_i][1:3])
                        self.robot.robot.P=opt_P
                        self.robot.robot.H=opt_H
                    robt_T = self.robot.fwd(rob_js_exe[scan_i][:6],world=True) # T_world^r2tool
                else:
                    robt_T = self.robot.fwd(rob_js_exe[scan_i][:6],world=True) # T_world^r2tool
                T_origin = self.positioner.fwd(rob_js_exe[scan_i][6:],world=True).inv() # T_tabletool^world
            T_rob_positioner_top = T_origin*robt_T

            if flip:
                scan_points=deepcopy(all_scan_points[scan_i].T)
            else:
                scan_points=deepcopy(all_scan_points[scan_i])
            if scanner=='mti':
                scan_points = np.insert(scan_points,1,np.zeros(len(scan_points[0])),axis=0)
                scan_points[0]=scan_points[0]*-1 # reversed x-axis
            else:
                scan_points = np.insert(scan_points,0,np.zeros(len(scan_points[0])),axis=0)
            
            
            scan_points = scan_points.T
            ## get the points closed to origin
            scan_points = np.transpose(np.matmul(T_rob_positioner_top.R,np.transpose(scan_points)))+T_rob_positioner_top.p
            # use legacy
            pcd = o3d.geometry.PointCloud()
            pcd.points=o3d.utility.Vector3dVector(scan_points)
            
            ## voxel down sample
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            # if scan_i==100:
            #     visualize_pcd([pcd])

            if pcd_combined is None:
                pcd_combined=deepcopy(pcd)
            else:
                pcd_combined+=pcd
        
        return pcd_combined
    
    def pcd_noise_remove(self,pcd_combined,voxel_down_flag=True,voxel_size=0.1,crop_flag=True,min_bound=(-50,-30,-10),max_bound=(50,30,50),\
                         outlier_remove=True,nb_neighbors=40,std_ratio=0.5,cluster_based_outlier_remove=True,cluster_neighbor=0.75,min_points=50*4):

        # visualize_pcd([pcd_combined])
        ## crop point clouds
        if crop_flag:
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
            pcd_combined=pcd_combined.crop(bbox)
        # visualize_pcd([pcd_combined])

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
            pcd_combined=pcd_combined.select_by_index(np.argwhere(labels>=0))
            print("Cluster based Outlier Removal done.")
        
        return pcd_combined

    def pcd2dh_compare(self,scanned_points,last_scanned_points,curve_relative,robot_weld=None,q_weld=None,ph_param=None,drawing=False):

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

        if robot_weld is not None:
            origin_P = deepcopy(robot_weld.robot.P)
            origin_H = deepcopy(robot_weld.robot.H)
            
            curve_relative = []
            for q in q_weld:
                Table_home_T = self.positioner.fwd(q[-2:])
                T_S1TCP_R1Base = np.matmul(self.positioner.base_H,H_from_RT(Table_home_T.R,Table_home_T.p))
                T_R1Base_S1TCP = np.linalg.inv(T_S1TCP_R1Base)
                
                ### R1 fwd
                if ph_param is not None:
                    opt_P,opt_H = ph_param.predict(q[1:3])
                    robot_weld.robot.P=opt_P
                    robot_weld.robot.H=opt_H
                robot_T = robot_weld.fwd(q[:6])
                ###
                
                T_R1TCP_S1TCP = np.matmul(T_R1Base_S1TCP,H_from_RT(robot_T.R,robot_T.p))
                curve_relative.append(np.append(T_R1TCP_S1TCP[:3,-1],T_R1TCP_S1TCP[:3,2]))
            
            robot_weld.robot.P=deepcopy(origin_P)
            robot_weld.robot.H=deepcopy(origin_H)
        
        curve_relative=np.array(curve_relative)

        # create the cropping polygon
        bounding_polygon=[]
        radius_scale=0.55
        # radius_scale=0.2
        radius=np.mean(np.linalg.norm(np.diff(curve_relative[:,:3],axis=0),axis=1))*radius_scale
        print("height neighbor radius:",radius)
        
        # circle        
        # poly_num=12
        # for n in range(poly_num):
        #     ang=(n/poly_num)*(np.pi*2)
        #     bounding_polygon.append(np.array([radius*np.cos(ang),radius*np.sin(ang),0]))
        # rectangle
        y_max=7
        bounding_polygon.append(np.array([radius,y_max,0]))
        bounding_polygon.append(np.array([radius,-y_max,0]))
        bounding_polygon.append(np.array([-radius,-y_max,0]))
        bounding_polygon.append(np.array([-radius,y_max,0]))
        ###
        
        bounding_polygon = np.array(bounding_polygon).astype("float64")
        crop_poly = o3d.visualization.SelectionPolygonVolume()
        crop_poly.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)
        crop_poly.orthogonal_axis = 'z'
        crop_poly.axis_max=30
        # crop_poly.axis_min=-15
        crop_poly.axis_min=-10

        if drawing:
            scanned_points_draw = deepcopy(scanned_points)
            scanned_points_draw.paint_uniform_color([0.3,0.3,0.3])
            last_scanned_points_draw = deepcopy(last_scanned_points)
            last_scanned_points_draw.paint_uniform_color([0.5,0.5,0.5])
            path_points = o3d.geometry.PointCloud()
            last_path_points = o3d.geometry.PointCloud()
            curve_R = []
            curve_p = []
        
        # loop through curve to get dh
        curve_i=0
        total_curve_i = len(curve_relative)
        dh=[]
        z_height=[]
        for curve_wp in curve_relative:
            if np.all(curve_wp==curve_relative[-1]):
                wp_R = direction2R_x(-1*curve_wp[3:],curve_wp[:3]-curve_relative[curve_i-1][:3])
            else:
                wp_R = direction2R_x(-1*curve_wp[3:],curve_relative[curve_i+1][:3]-curve_wp[:3])

            sp_lamx=deepcopy(scanned_points)
            ## transform the scanned points to waypoints
            sp_lamx.transform(np.linalg.inv(H_from_RT(wp_R,curve_wp[:3])))
            # visualize_pcd([sp_lamx])
            ## crop the scanned points around the waypoints
            sp_lamx = crop_poly.crop_point_cloud(sp_lamx)
            # visualize_pcd([sp_lamx],origin_size=10)
            ## dh is simply the z height after transform. Average over an radius
            
            last_sp_lamx=deepcopy(last_scanned_points)
            last_sp_lamx.transform(np.linalg.inv(H_from_RT(wp_R,curve_wp[:3])))
            last_sp_lamx = crop_poly.crop_point_cloud(last_sp_lamx)
            
            percentage=0.05
            this_points_z = np.asarray(sp_lamx.points)[:,2]
            if len(this_points_z>0):
                height_ids=max(int(percentage*len(this_points_z)),10)
                this_points_z = np.sort(this_points_z)[-1*height_ids:]
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1e5,-1e5,this_points_z[0]),max_bound=(1e5,1e5,1e5))
                sp_lamx=sp_lamx.crop(bbox)
                
                # visualize_pcd([sp_lamx],origin_size=2)

                last_points_z = np.asarray(last_sp_lamx.points)[:,2]
                if len(last_points_z>0):
                    height_ids=max(int(percentage*len(last_points_z)),10)
                    last_points_z = np.sort(last_points_z)[-1*height_ids:]
                    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1e5,-1e5,last_points_z[0]),max_bound=(1e5,1e5,1e5))
                    last_sp_lamx=last_sp_lamx.crop(bbox)
                else:
                    print("not find")
                    this_points_z=np.nan
                    last_points_z=np.nan
            else:
                print("not find")
                this_points_z=np.nan
                last_points_z=np.nan
            
            this_dh = np.mean(this_points_z)-np.mean(last_points_z)
            this_zheight = np.mean(this_points_z)

            dh_max=7
            dh_min=-2
            this_dh = max(min(this_dh,dh_max),dh_min)
            # if this_dh>dh_max:
            #     this_dh=np.nan

            dh.append(this_dh)
            z_height.append(this_zheight)

            if drawing:
                ## paint pcd for visualization
                color_dist = plt.get_cmap("rainbow")(float(curve_i)/total_curve_i)
                sp_lamx.paint_uniform_color(color_dist[:3])
                sp_lamx.transform(H_from_RT(wp_R,curve_wp[:3]))
                path_points = path_points+sp_lamx
                last_sp_lamx.paint_uniform_color(color_dist[:3])
                last_sp_lamx.transform(H_from_RT(wp_R,curve_wp[:3]))
                last_path_points = last_path_points+last_sp_lamx
                curve_R.append(wp_R)
                curve_p.append(curve_wp[:3])

            curve_i+=1

        window_nan=3
        for curve_i in range(len(dh)):
            if np.isnan(dh[curve_i]):
                if curve_i<window_nan:
                    dh[curve_i]=np.nanmean(dh[0:2*window_nan])
                elif curve_i>len(dh)-window_nan:
                    dh[curve_i]=np.nanmean(dh[-2*window_nan:])
                else:
                    dh[curve_i]=np.nanmean(dh[curve_i-window_nan:curve_i+window_nan])
            if np.isnan(z_height[curve_i]):
                if curve_i<window_nan:
                    z_height[curve_i]=np.nanmean(z_height[0:2*window_nan])
                elif curve_i>len(z_height)-window_nan:
                    z_height[curve_i]=np.nanmean(z_height[-2*window_nan:])
                else:
                    z_height[curve_i]=np.nanmean(z_height[curve_i-window_nan:curve_i+window_nan])
        # input(dh)

        curve_relative=np.array(curve_relative)
        lam = calc_lam_cs(curve_relative[:,:3])
        profile_dh = np.array([lam,dh]).T 
        profile_height = np.array([lam,z_height]).T   

        if drawing:
            path_points.transform(H_from_RT(np.eye(3),[0,0,0.0001]))
            last_path_points.transform(H_from_RT(np.eye(3),[0,0,0.0001]))
            path_viz_frames = visualize_frames(curve_R,curve_p,size=1,visualize=False,frame_obj=True)
            draw_obj = []
            draw_obj.extend(path_viz_frames)
            draw_obj.extend([scanned_points_draw,path_points,last_scanned_points_draw,last_path_points])
            visualize_pcd(draw_obj)
        
        return profile_dh,profile_height
    
    def pcd2dh(self,scanned_points,curve_relative,robot_weld=None,q_weld=None,ph_param=None,drawing=False):

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

        if robot_weld is not None:
            origin_P = deepcopy(robot_weld.robot.P)
            origin_H = deepcopy(robot_weld.robot.H)
            
            curve_relative = []
            for q in q_weld:
                Table_home_T = self.positioner.fwd(q[-2:])
                T_S1TCP_R1Base = np.matmul(self.positioner.base_H,H_from_RT(Table_home_T.R,Table_home_T.p))
                T_R1Base_S1TCP = np.linalg.inv(T_S1TCP_R1Base)
                
                ### R1 fwd
                if ph_param is not None:
                    opt_P,opt_H = ph_param.predict(q[1:3])
                    robot_weld.robot.P=opt_P
                    robot_weld.robot.H=opt_H
                robot_T = robot_weld.fwd(q[:6])
                ###
                
                T_R1TCP_S1TCP = np.matmul(T_R1Base_S1TCP,H_from_RT(robot_T.R,robot_T.p))
                curve_relative.append(np.append(T_R1TCP_S1TCP[:3,-1],T_R1TCP_S1TCP[:3,2]))
            
            robot_weld.robot.P=deepcopy(origin_P)
            robot_weld.robot.H=deepcopy(origin_H)
        
        curve_relative=np.array(curve_relative)

        # create the cropping polygon
        bounding_polygon=[]
        radius_scale=0.8
        # radius_scale=0.2
        radius=np.mean(np.linalg.norm(np.diff(curve_relative[:,:3],axis=0),axis=1))*radius_scale
        print("height neighbor radius:",radius)
        
        # circle        
        # poly_num=12
        # for n in range(poly_num):
        #     ang=(n/poly_num)*(np.pi*2)
        #     bounding_polygon.append(np.array([radius*np.cos(ang),radius*np.sin(ang),0]))
        # rectangle
        y_max=5
        bounding_polygon.append(np.array([radius,y_max,0]))
        bounding_polygon.append(np.array([radius,-y_max,0]))
        bounding_polygon.append(np.array([-radius,-y_max,0]))
        bounding_polygon.append(np.array([-radius,y_max,0]))
        ###
        
        bounding_polygon = np.array(bounding_polygon).astype("float64")
        crop_poly = o3d.visualization.SelectionPolygonVolume()
        crop_poly.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)
        crop_poly.orthogonal_axis = 'z'
        crop_poly.axis_max=30
        # crop_poly.axis_min=-15
        crop_poly.axis_min=-5

        if drawing:
            scanned_points_draw = deepcopy(scanned_points)
            scanned_points_draw.paint_uniform_color([0.3,0.3,0.3])
            path_points = o3d.geometry.PointCloud()
            curve_R = []
            curve_p = []
        
        # loop through curve to get dh
        curve_i=0
        total_curve_i = len(curve_relative)
        dh=[]
        for curve_wp in curve_relative:
            if np.all(curve_wp==curve_relative[-1]):
                wp_R = direction2R_x(-1*curve_wp[3:],curve_wp[:3]-curve_relative[curve_i-1][:3])
            else:
                wp_R = direction2R_x(-1*curve_wp[3:],curve_relative[curve_i+1][:3]-curve_wp[:3])

            sp_lamx=deepcopy(scanned_points)
            ## transform the scanned points to waypoints
            sp_lamx.transform(np.linalg.inv(H_from_RT(wp_R,curve_wp[:3])))
            # visualize_pcd([sp_lamx])
            ## crop the scanned points around the waypoints
            sp_lamx = crop_poly.crop_point_cloud(sp_lamx)
            # visualize_pcd([sp_lamx],origin_size=10)
            ## dh is simply the z height after transform. Average over an radius
            
            percentage=0.05
            this_points_z = np.asarray(sp_lamx.points)[:,2]
            if len(this_points_z>0):
                height_ids=max(int(percentage*len(this_points_z)),10)
                this_points_z = np.sort(this_points_z)[-1*height_ids:]
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1e5,-1e5,this_points_z[0]),max_bound=(1e5,1e5,1e5))
                sp_lamx=sp_lamx.crop(bbox)
            else:
                print("not find")
                this_points_z=np.nan
            
            
            this_dh = np.nanmean(this_points_z)

            dh_max=10
            dh_min=-10
            this_dh = max(min(this_dh,dh_max),dh_min)
            # if this_dh>dh_max:
            #     this_dh=np.nan

            dh.append(this_dh)

            if drawing:
                ## paint pcd for visualization
                color_dist = plt.get_cmap("rainbow")(float(curve_i)/total_curve_i)
                sp_lamx.paint_uniform_color(color_dist[:3])
                sp_lamx.transform(H_from_RT(wp_R,curve_wp[:3]))
                path_points = path_points+sp_lamx
                curve_R.append(wp_R)
                curve_p.append(curve_wp[:3])

            curve_i+=1

        window_nan=3
        for curve_i in range(len(dh)):
            if np.isnan(dh[curve_i]):
                if curve_i<window_nan:
                    dh[curve_i]=np.nanmean(dh[0:2*window_nan])
                elif curve_i>len(dh)-window_nan:
                    dh[curve_i]=np.nanmean(dh[-2*window_nan:])
                else:
                    dh[curve_i]=np.nanmean(dh[curve_i-window_nan:curve_i+window_nan])
        # input(dh)

        curve_relative=np.array(curve_relative)
        lam = calc_lam_cs(curve_relative[:,:3])
        profile_height = np.array([lam,dh]).T   

        if drawing:
            path_points.transform(H_from_RT(np.eye(3),[0,0,0.0001]))
            path_viz_frames = visualize_frames(curve_R,curve_p,size=1,visualize=False,frame_obj=True)
            draw_obj = []
            draw_obj.extend(path_viz_frames)
            draw_obj.extend([scanned_points_draw,path_points])
            visualize_pcd(draw_obj)
        
        return profile_height
    
    def dh2height(self,layer_curve_relative,layer_curve_dh,last_curve_relative,last_curve_height):
        
        last_curve_relative=np.array(last_curve_relative)
        layer_curve_height=[]
        for this_id in range(len(layer_curve_relative)):
            this_p = layer_curve_relative[this_id][:3]
            last_p_id = np.argmin(np.linalg.norm(last_curve_relative[:,:3]-this_p,2,1))
            p_height = last_curve_height[last_p_id]+layer_curve_dh[this_id][1]
            layer_curve_height.append(p_height)
        
        return layer_curve_height
    
    def pcd_calib_z(self,scanned_points,Transz0_H=None):
        
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
        return scanned_points,Transz0_H
    
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

    def scan2dh(self,scan,robot_q,target_p,crop_min=[-10,85],crop_max=[10,100],offset_z=2.2):
        
        dbscan = DBSCAN(eps=0.5,min_samples=20)
        ## remove not in interested region
        st=time.time()
        mti_pcd=np.delete(scan,scan[1]==0,axis=1)
        mti_pcd=np.delete(mti_pcd,mti_pcd[1]<crop_min[1],axis=1)
        mti_pcd=np.delete(mti_pcd,mti_pcd[1]>crop_max[1],axis=1)
        mti_pcd=np.delete(mti_pcd,mti_pcd[0]<crop_min[0],axis=1)
        mti_pcd=np.delete(mti_pcd,mti_pcd[0]>crop_max[0],axis=1)
        mti_pcd[0]=-1*mti_pcd[0]
        mti_pcd = mti_pcd.T
        
        # cluster based noise remove
        dbscan.fit(mti_pcd)
        n_clusters_ = len(set(dbscan.labels_))

        if n_clusters_>1:
            cluster_id = dbscan.labels_>=0
            mti_pcd_noise_remove=mti_pcd[cluster_id]
        else:
            mti_pcd_noise_remove=mti_pcd
        
        # transform to R2TCP
        T_R2TCP_S1TCP=self.positioner.fwd(robot_q[6:],world=True).inv()*self.robot.fwd(robot_q[:6],world=True)
        target_z = np.array(target_p)
        largest_id = np.argsort(mti_pcd_noise_remove[:,1])[:10]
        point_location = np.mean(mti_pcd_noise_remove[largest_id],axis=0)
        point_location_z_R2TCP = deepcopy(point_location[1])
        point_location=np.insert(point_location,1,0)
        point_location = np.matmul(T_R2TCP_S1TCP.R,point_location)+T_R2TCP_S1TCP.p
        point_location[2]=point_location[2]-offset_z

        delta_h = (target_z[2]-point_location[2])

        # for cluster_i in range(n_clusters_-1):
        #     cluster_id = dbscan.labels_==cluster_i
        #     plt.scatter(-1*mti_pcd[cluster_id][:,0],mti_pcd[cluster_id][:,1])
        # plt.scatter(-1*mti_pcd[:,0],mti_pcd[:,1])
        # # plt.axhline(y = target_z[2], color = 'r', linestyle = '-')
        # plt.axhline(y = point_location_z_R2TCP-delta_h, color = 'r', linestyle = '-')
        # plt.xlim((-30,30))
        # # plt.ylim((50,120))
        # plt.ylim((0,120))
        # plt.show()

        return delta_h,point_location