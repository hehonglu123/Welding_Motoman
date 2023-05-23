import open3d as o3d

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import time
from copy import deepcopy
import colorsys
import math
import pickle

def icp_align2(source_points,target_points,H=np.eye(4),icp_turns = 1,threshold=5,max_iteration=1000):
    ###find transformation from pc1 to pc2 with ICP, with initial guess H
    # Convert numpy arrays to Open3D point cloud format
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source_points)

    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target_points)

    for i in range(icp_turns):
        icp_iteration = o3d.pipelines.registration.registration_icp(
            source_cloud, target_cloud,init=H, max_correspondence_distance=1.5,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1.000000e-06, relative_rmse=1.000000e-06, max_iteration=max_iteration))
        H = icp_iteration.transformation
        print(icp_iteration)
    return H


def icp_align(pc1,pc2,H=np.eye(4),icp_turns = 1,threshold=5,max_iteration=1000):
    ###find transformation from pc1 to pc2 with ICP, with initial guess H
    pc1_o3d=o3d.geometry.PointCloud()
    pc1_o3d.points = o3d.utility.Vector3dVector(pc1)
    pc2_o3d=o3d.geometry.PointCloud()
    pc2_o3d.points = o3d.utility.Vector3dVector(pc2)
    for i in range(icp_turns):
        reg_p2p = o3d.pipelines.registration.registration_icp(
                    pc1_o3d, pc2_o3d, threshold, H,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
        print(reg_p2p)
        pc1_o3d=pc1_o3d.transform(reg_p2p.transformation)
        H = reg_p2p.transformation@H
    return H

def colormap(all_h):

    all_h=(1-all_h)*270
    all_color=[]
    for h in all_h:
        all_color.append(colorsys.hsv_to_rgb(h,0.7,0.9))
    return np.array(all_color)

def display_inlier_outlier(cloud, ind):
    
    if type(cloud) is o3d.cpu.pybind.t.geometry.PointCloud:
        cloud_show = cloud.to_legacy()
    else:
        cloud_show=cloud

    points_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20,origin=[0,0,0])
    inlier_cloud = cloud_show.select_by_index(ind)
    outlier_cloud = cloud_show.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0, 0.8, 0])
    show_pcd_list=[inlier_cloud, outlier_cloud, points_frame]
    o3d.visualization.draw_geometries(show_pcd_list,width=960,height=540)

def visualize_pcd(show_pcd_list,point_show_normal=False):

    show_pcd_list_legacy=[]
    for obj in show_pcd_list:
        if type(obj) is o3d.cpu.pybind.t.geometry.PointCloud or type(obj) is o3d.cpu.pybind.t.geometry.TriangleMesh:
            show_pcd_list_legacy.append(obj.to_legacy())
        else:
            show_pcd_list_legacy.append(obj)

    points_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20,origin=[0,0,0])
    show_pcd_list_legacy.append(points_frame)
    o3d.visualization.draw_geometries(show_pcd_list_legacy,width=960,height=540,point_show_normal=point_show_normal)

def visualize_frames(all_R,all_p,size=20):

    all_points_frame = []
    for i in range(len(all_R)):
        points_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        points_frame = points_frame.rotate(all_R[i],center=[0,0,0])
        points_frame = points_frame.translate(all_p[i])
        
        all_points_frame.append(points_frame)
    o3d.visualization.draw_geometries(all_points_frame,width=960,height=540)