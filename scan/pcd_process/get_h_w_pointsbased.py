from cProfile import label
import sys
import matplotlib

sys.path.append('../../toolbox/')
from robot_def import *
from utils import *
from general_robotics_toolbox import *
import open3d as o3d

import numpy as np
import cv2
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
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0, 0.8, 0])
    o3d.visualization.draw([inlier_cloud, outlier_cloud])

def visualize_pcd(show_pcd_list,point_show_normal=False):
	points_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3,origin=[0,0,0])
	show_pcd_list.append(points_frame)
	o3d.visualization.draw_geometries(show_pcd_list,width=960,height=540,point_show_normal=point_show_normal)
	# o3d.visualization.draw(show_pcd_list,width=960,height=540)

data_dir='../../data/wall_weld_test/test3_2/'
config_dir='../../config/'


######## read the combined mesh
scanned_points = o3d.io.read_point_cloud(data_dir+'processed_pcd.pcd')
# visualize_pcd([scanned_points])
####### plane segmentation

plane_model, inliers = scanned_points.segment_plane(distance_threshold=0.75,
                                         ransac_n=5,
                                         num_iterations=3000)
# print(plane_model)
# display_inlier_outlier(scanned_points,inliers)
## Transform the plane to z=0
Transz0 = Transform(rot(np.cross(plane_model[:3],[0,0,1]),np.arccos(plane_model[2])),[0,0,0])*\
			Transform(np.eye(3),[0,0,plane_model[3]/plane_model[2]])
Transz0_H=H_from_RT(Transz0.R,Transz0.p)
scanned_points.transform(Transz0_H)
# plane_model, inliers = scanned_points.segment_plane(distance_threshold=0.75,
#                                          ransac_n=5,
#                                          num_iterations=3000)
# print(plane_model)
# display_inlier_outlier(scanned_points,inliers)
# visualize_pcd([scanned_points])
### now the distance to plane is the z axis

## Transform such that the path is in y-axis
Trans_zaxis=np.eye(4)
Trans_zaxis[:3,:3]=rot([0,0,1],np.radians(1))
scanned_points.transform(Trans_zaxis)

# y-axis box
y_axis_mesh = o3d.geometry.TriangleMesh.create_box(width=0.1, height=200.0, depth=10)
box_move=np.eye(4)
box_move[0,3]=86
box_move[1,3]=-200
box_move[2,3]=-5
y_axis_mesh.transform(box_move)

# bbox for each weld
# z_axis_mesh = o3d.geometry.TriangleMesh.create_box(width=20, height=85, depth=0.1)
# box_move=np.eye(4)
# box_move[0,3]=104
# box_move[1,3]=-116
# box_move[2,3]=0
# z_axis_mesh.transform(box_move)

bbox_1_min=(-3,-115,-10)
bbox_1_max=(17,-30,100)
bbox_2_min=(21,-116,-10)
bbox_2_max=(41,-31,100)
bbox_3_min=(50,-116,-10)
bbox_3_max=(70,-31,100)
bbox_4_min=(77,-116,-10)
bbox_4_max=(97,-31,100)
bbox_5_min=(104,-116,-10)
bbox_5_max=(124,-31,100)

boxes_min=[bbox_1_min,bbox_2_min,bbox_3_min,bbox_4_min,bbox_5_min]
boxes_max=[bbox_1_max,bbox_2_max,bbox_3_max,bbox_4_max,bbox_5_max]

bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1e5,-118,-1e5),max_bound=(1e5,-29,1e5))
scanned_points=scanned_points.crop(bbox)
visualize_pcd([scanned_points])
##### plot
plot_flag=True

##### store cross section data
all_welds_width=[]
all_welds_height=[]
for weld_i in range(len(boxes_min)):
    all_welds_width.append({})
    all_welds_height.append({})

##### cross section parameters
resolution_z=0.1
windows_z=0.2
resolution_y=0.1
windows_y=1
stop_thres=20
stop_thres_w=10
use_points_num=5 # use the largest/smallest N to compute w
width_thres=0.8 # prune width that is too close
all_y_min=[]
all_y_max=[]

##### get projection of each z height
z_max=np.max(np.asarray(scanned_points.points)[:,2])
for z in np.arange(1.5,z_max+resolution_z,resolution_z):
    print(z)
    #### crop z height
    min_bound = (-1e5,-1e5,z-windows_z/2)
    max_bound = (1e5,1e5,z+windows_z/2)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
    points_proj=scanned_points.crop(bbox)
    ##################

    #### plot w h
    f, (axw, axh) = plt.subplots(2, 1, sharex=True)
    #### crop welds
    all_welds_points = o3d.geometry.PointCloud()
    for weld_i in range(len(boxes_min)):
        print('Weld i',weld_i)
        min_bound = boxes_min[weld_i]
        max_bound = boxes_max[weld_i]
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
        welds_points=points_proj.crop(bbox)

        #### get width with y-direction scanning
        if len(welds_points.points)<stop_thres:
            continue
        all_welds_width[weld_i][z]={}
        all_welds_height[weld_i][z]={}

        if len(all_y_min)<=weld_i:
            all_y_min.append(boxes_min[weld_i][1]+resolution_y)
            all_y_max.append(boxes_max[weld_i][1]-resolution_y)
        for y in np.arange(all_y_min[weld_i],all_y_max[weld_i]+resolution_y,resolution_y):
            min_bound = (-1e5,y-windows_y/2,-1e5)
            max_bound = (1e5,y+windows_y/2,1e5)
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
            welds_points_y = welds_points.crop(bbox)
            if len(welds_points_y.points)<stop_thres_w:
                # all_welds_width[weld_i][z][y]=0
                # all_welds_height[weld_i][z][y]=0
                continue
            ### get the width
            sort_x=np.argsort(np.asarray(welds_points_y.points)[:,0])
            x_min_index=sort_x[:use_points_num]
            x_max_index=sort_x[-use_points_num:]
            
            ### prune x that is too closed
            x_min_all = np.asarray(welds_points_y.points)[x_min_index,0]
            x_min = np.mean(x_min_all)
            x_max_all = np.asarray(welds_points_y.points)[x_max_index,0]
            x_max = np.mean(x_max_all)

            actual_x_min_all=[]
            actual_x_max_all=[]
            for num_i in range(use_points_num):
                if (x_max-x_min_all[num_i])>width_thres:
                    actual_x_min_all.append(x_min_all[num_i])
                if (x_max_all[num_i]-x_min)>width_thres:
                    actual_x_max_all.append(x_max_all[num_i])
            #########
            x_max=0
            x_min=0
            if len(actual_x_max_all)!=0 and len(actual_x_min_all)!=0:
                x_max=np.mean(actual_x_max_all)
                x_min=np.mean(actual_x_min_all)

            this_width=x_max-x_min
            all_welds_width[weld_i][z][y]=this_width
            z_height_ave = np.mean(np.asarray(welds_points_y.points)[np.append(x_min_index,x_max_index),2])
            all_welds_height[weld_i][z][y]=z_height_ave
            # visualize_pcd([welds_points_y])
        ### get all zy coord
        y_coord=np.array(list(all_welds_width[weld_i][z].keys()))
        y_width=np.array(list(all_welds_width[weld_i][z].values()))
        y_height=np.array(list(all_welds_height[weld_i][z].values()))
        if plot_flag:
            ### plot width and height
            axw.plot(y_coord,y_width,marker='o',color=table_colors[weld_i],label='Weld Number '+str(weld_i))
            axh.plot(y_coord,y_height,marker='o',color=table_colors[weld_i],label='Weld Number '+str(weld_i))
        
        welds_points.paint_uniform_color(mcolors.to_rgb(table_colors[weld_i]))
        all_welds_points+=welds_points

    if plot_flag:
        # points_proj.paint_uniform_color([1, 0, 0])
        # welds_points.paint_uniform_color([0, 0.8, 0])
        visualize_pcd([all_welds_points])
        axw.set_ylim([0, axw.get_ylim()[1]+axw.get_ylim()[1]/5])
        axw.tick_params(axis="x", labelsize=14) 
        axw.tick_params(axis="y", labelsize=14) 
        axw.set_ylabel('width (mm)',fontsize=16)
        axh.set_ylim([0, axh.get_ylim()[1]+axh.get_ylim()[1]/5])
        axh.tick_params(axis="x", labelsize=14) 
        axh.tick_params(axis="y", labelsize=14) 
        axh.set_ylabel('height (mm)',fontsize=16)
        plt.xlabel('y-axis (mm)',fontsize=16)
        plt.legend()
        plt.show()
    # exit()

pickle.dump(all_welds_width, open(data_dir+'all_welds_width.pickle','wb'))
pickle.dump(all_welds_height, open(data_dir+'all_welds_height.pickle','wb'))