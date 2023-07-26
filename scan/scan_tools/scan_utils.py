from cProfile import label
import sys
import matplotlib

sys.path.append('../../toolbox/')
from robot_def import *
from utils import *
from general_robotics_toolbox import *
import open3d as o3d

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import time
from copy import deepcopy
import colorsys
import math
import pickle

table_colors = list(mcolors.TABLEAU_COLORS.values())

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

def visualize_pcd(show_pcd_list,point_show_normal=False,origin_size=20):

    show_pcd_list_legacy=[]
    for obj in show_pcd_list:
        if type(obj) is o3d.cpu.pybind.t.geometry.PointCloud or type(obj) is o3d.cpu.pybind.t.geometry.TriangleMesh:
            show_pcd_list_legacy.append(obj.to_legacy())
        else:
            show_pcd_list_legacy.append(obj)

    points_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=origin_size,origin=[0,0,0])
    show_pcd_list_legacy.append(points_frame)
    o3d.visualization.draw_geometries(show_pcd_list_legacy,width=960,height=540,point_show_normal=point_show_normal)

def visualize_frames(all_R,all_p,size=20,visualize=True,frame_obj=False):

    all_points_frame = []
    for i in range(len(all_R)):
        points_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        points_frame = points_frame.rotate(all_R[i],center=[0,0,0])
        points_frame = points_frame.translate(all_p[i])
        
        all_points_frame.append(points_frame)
            
    if visualize:
        o3d.visualization.draw_geometries(all_points_frame,width=960,height=540)
    if frame_obj:
        return all_points_frame