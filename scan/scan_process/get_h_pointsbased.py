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
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import time
from copy import deepcopy
import colorsys
import math
import pickle

table_colors = list(mcolors.TABLEAU_COLORS.values())

# data_dir='../../data/wall_weld_test/top_layer_test/scans/'
data_dir='../../data/wall_weld_test/top_layer_test_mti/scans/'
config_dir='../../config/'

######## read the combined point clouds
scanned_points = o3d.io.read_point_cloud(data_dir+'processed_pcd.pcd')
print(len(scanned_points.points))

# visualize_pcd([scanned_points])

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
### now the distance to plane is the z axis

# visualize_pcd([scanned_points])

## TODO:align path and scan

# bbox for each weld
bbox_mesh = o3d.geometry.TriangleMesh.create_box(width=80, height=20, depth=0.1)
box_move=np.eye(4)
box_move[0,3]=-40
box_move[1,3]=-10
box_move[2,3]=0
bbox_mesh.transform(box_move)

bbox_min=(-40,-20,0)
bbox_max=(40,20,45)

# visualize_pcd([scanned_points,bbox_mesh])
##################### get welding pieces end ########################


##### plot
plot_flag=False

##### cross section parameters
z_height_start=35
resolution_z=0.1
windows_z=0.2
resolution_x=0.1
windows_x=1
stop_thres=20
stop_thres_w=10
use_points_num=5 # use the largest/smallest N to compute w
width_thres=0.8 # prune width that is too close

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

    #### crop welds
    all_welds_points = o3d.geometry.PointCloud()
    
    min_bound = bbox_min
    max_bound = bbox_max
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
    welds_points=points_proj.crop(bbox)

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
        z_height_ave = np.mean(np.asarray(welds_points_x.points)[np.append(y_min_index,y_max_index),2])
        profile_p.append(np.array([x,this_width,z_height_ave]))
    profile_p = np.array(profile_p)
    if plot_flag:
        visualize_pcd([welds_points])
        ### plot width and height
        plt.plot(profile_p[:,0],profile_p[:,2],'-o')
        plt.show()
    
    for pf_i in range(len(profile_p)):
        profile_height[profile_p[pf_i][0]] = profile_p[pf_i][2]

    # exit()
profile_height_arr = []
for x in profile_height.keys():
    profile_height_arr.append(np.array([x,profile_height[x]]))
profile_height_arr=np.array(profile_height_arr)

profile_height_arr_argsort = np.argsort(profile_height_arr[:,0])
profile_height_arr=profile_height_arr[profile_height_arr_argsort]

# plt.scatter(profile_height_arr[:,0],profile_height_arr[:,1])
# plt.show()

profile_height=profile_height_arr
profile_slope = np.diff(profile_height[:,1])/np.diff(profile_height[:,0])

plt.scatter(profile_height_arr[:,0],profile_height_arr[:,1]-np.mean(profile_height_arr[:,1]))
plt.plot(profile_height[1:,0],profile_slope)
plt.show()



# pickle.dump(all_welds_width, open(data_dir+'all_welds_width.pickle','wb'))
# pickle.dump(all_welds_height, open(data_dir+'all_welds_height.pickle','wb'))