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
    points_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20,origin=[0,0,0])
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0, 0.8, 0])
    show_pcd_list=[inlier_cloud, outlier_cloud, points_frame]
    o3d.visualization.draw_geometries(show_pcd_list,width=960,height=540)

def visualize_pcd(show_pcd_list,point_show_normal=False):
	points_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20,origin=[0,0,0])
	show_pcd_list.append(points_frame)
	o3d.visualization.draw_geometries(show_pcd_list,width=960,height=540,point_show_normal=point_show_normal)

def visualize_frames(all_R,all_p,size=20):

	all_points_frame = []
	for i in range(len(all_R)):
		points_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
		points_frame = points_frame.rotate(all_R[i],center=[0,0,0])
		points_frame = points_frame.translate(all_p[i])
		
		all_points_frame.append(points_frame)
	o3d.visualization.draw_geometries(all_points_frame,width=960,height=540)