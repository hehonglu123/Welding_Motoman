from cProfile import label
import sys
import matplotlib

sys.path.append('../../toolbox/')
sys.path.append('../scan_tools')
from robot_def import *
from utils import *
from scan_utils import *
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

data_dir='../../data/wall_weld_test/test4/150_ipm_1st_layer/'
config_dir='../../config/'

######## read the combined point clouds
scanned_points_mesh = o3d.io.read_triangle_mesh(data_dir+'processed_pcd.stl')

scanned_points = scanned_points_mesh.sample_points_uniformly(number_of_points=1110000)
visualize_pcd([scanned_points_mesh,scanned_points])

###################### get the welding pieces ##################
# This part will be replaced by welding path in the future
######## make the plane normal as z-axis
####### plane segmentation
plane_model, inliers = scanned_points.segment_plane(distance_threshold=0.75,
                                         ransac_n=5,
                                         num_iterations=3000)
# display_inlier_outlier(scanned_points,inliers)
## Transform the plane to z=0

plain_norm = plane_model[:3]/np.linalg.norm(plane_model[:3])
k = np.cross(plain_norm,[0,0,1])
k = k/np.linalg.norm(k)
theta = np.arccos(plain_norm[2])
Transz0 = Transform(rot(k,theta),[0,0,0])*\
			Transform(np.eye(3),[0,0,plane_model[3]/plane_model[2]])
Transz0_H=H_from_RT(Transz0.R,Transz0.p)
scanned_points.transform(Transz0_H)
# visualize_pcd([scanned_points])
### now the distance to plane is the z axis

# x-axis box
x_axis_mesh = o3d.geometry.TriangleMesh.create_box(width=200, height=0.1, depth=10)
box_move=np.eye(4)
box_move[0,3]=0
box_move[1,3]=0
box_move[2,3]=-5
x_axis_mesh.transform(box_move)

## Transform such that the path is in x-axis
rotation_theta=np.radians(-1) ## rotation angle such that path align x-axis
translation_p = np.array([0,0,0]) ## Translation is less matters here
Trans_zaxis=np.eye(4)
Trans_zaxis[:3,:3]=rot([0,0,1],rotation_theta)
Trans_zaxis[:3,3]=translation_p
scanned_points.transform(Trans_zaxis)

# bbox for each weld
bbox_mesh = o3d.geometry.TriangleMesh.create_box(width=85, height=20, depth=0.1)
box_move=np.eye(4)
box_move[0,3]=-32 # x-axis
box_move[1,3]=-25 # y-axis
box_move[2,3]=1
bbox_mesh.transform(box_move)

bbox_1_min=(-32,-52,-10)
bbox_1_max=(53,-32,100)

# bbox_2_min=(1,-57,-10)
# bbox_2_max=(86,-37,100)


# bbox_3_min=(1,-85,-10)
# bbox_3_max=(86,-65,100)
# bbox_4_min=(0,-112,-10)
# bbox_4_max=(85,-92,100)

# boxes_min=[bbox_1_min,bbox_2_min,bbox_3_min,bbox_4_min]
# boxes_max=[bbox_1_max,bbox_2_max,bbox_3_max,bbox_4_max]
boxes_min=[bbox_1_min]
boxes_max=[bbox_1_max]

bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1e5,-1e5,-1e5),max_bound=(1e5,1e5,1e5))
scanned_points=scanned_points.crop(bbox)
visualize_pcd([scanned_points,x_axis_mesh,bbox_mesh])
##################### get welding pieces end ########################


##### plot
plot_flag=False

##### store cross section data
all_welds_width=[]
all_welds_height=[]
for weld_i in range(len(boxes_min)):
    all_welds_width.append({})
    all_welds_height.append({})

##### cross section parameters
z_height_start=0.5


resolution_z=0.1
windows_z=0.2
resolution_x=0.1
windows_x=1
stop_thres=20
stop_thres_w=10
use_points_num=5 # use the largest/smallest N to compute w
width_thres=0.8 # prune width that is too close
all_x_min=[]
all_x_max=[]

##### get projection of each z height
z_max=np.max(np.asarray(scanned_points.points)[:,2])
for z in np.arange(z_height_start,z_max+resolution_z,resolution_z):
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

        #### get width with x-direction scanning
        if len(welds_points.points)<stop_thres:
            continue
        all_welds_width[weld_i][z]={}
        all_welds_height[weld_i][z]={}

        if len(all_x_min)<=weld_i:
            all_x_min.append(boxes_min[weld_i][0])
            all_x_max.append(boxes_max[weld_i][0])
        for x in np.arange(all_x_min[weld_i],all_x_max[weld_i]+resolution_x,resolution_x):
            min_bound = (x-windows_x/2,-1e5,-1e5)
            max_bound = (x+windows_x/2,1e5,1e5)
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
            welds_points_x = welds_points.crop(bbox)
            if len(welds_points_x.points)<stop_thres_w:
                # all_welds_width[weld_i][z][x]=0
                # all_welds_height[weld_i][z][x]=0
                continue
            # visualize_pcd([welds_points_x])
            ### get the width
            sort_y=np.argsort(np.asarray(welds_points_x.points)[:,1])
            y_min_index=sort_y[:use_points_num]
            y_max_index=sort_y[-use_points_num:]
            
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
            all_welds_width[weld_i][z][x]=this_width
            z_height_ave = np.mean(np.asarray(welds_points_x.points)[np.append(y_min_index,y_max_index),2])
            all_welds_height[weld_i][z][x]=z_height_ave
        ### get all zx coord
        x_coord=np.array(list(all_welds_width[weld_i][z].keys()))
        x_width=np.array(list(all_welds_width[weld_i][z].values()))
        x_height=np.array(list(all_welds_height[weld_i][z].values()))
        if plot_flag:
            ### plot width and height
            axw.plot(x_coord,x_width,marker='o',color=table_colors[weld_i],label='Weld Number '+str(weld_i))
            axh.plot(x_coord,x_height,marker='o',color=table_colors[weld_i],label='Weld Number '+str(weld_i))
        
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
        plt.xlabel('x-axis (mm)',fontsize=16)
        plt.legend()
        plt.show()
    # exit()

pickle.dump(all_welds_width, open(data_dir+'all_welds_width.pickle','wb'))
pickle.dump(all_welds_height, open(data_dir+'all_welds_height.pickle','wb'))