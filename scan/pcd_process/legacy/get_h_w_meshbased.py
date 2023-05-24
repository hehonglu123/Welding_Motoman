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

#### create mesh
### parameters
density_remove=0.01
####### estimate normal
scanned_points.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# ####### estimate mesh
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    rec_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        scanned_points, depth=9)
## remove density less than
densities = np.asarray(densities)
density_colors = plt.get_cmap('plasma')(
    (densities - densities.min()) / (densities.max() - densities.min()))
density_colors = density_colors[:, :3]
density_mesh = o3d.geometry.TriangleMesh()
density_mesh.vertices = rec_mesh.vertices
density_mesh.triangles = rec_mesh.triangles
density_mesh.triangle_normals = rec_mesh.triangle_normals
density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
vertices_to_remove = densities < np.quantile(densities, density_remove)
density_mesh.remove_vertices_by_mask(vertices_to_remove)
# visualize_pcd([density_mesh])
rec_mesh=density_mesh
############################

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

##### store cross section data
all_welds=[]
for weld_i in range(len(boxes_min)):
    all_welds.append({})

##### cross section parameters
resolution_z=0.1
windows_z=0.2
resolution_y=0.1
windows_y=2
stop_thres=20
stop_thres_w=10
use_points_num=5 # use the largest/smallest N to compute w

## crop z smaller than 0
z_max=np.max(np.asarray(scanned_points.points)[:,2])
min_bound = (-1e5,-1e5,0)
max_bound = (1e5,1e5,z_max)
bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
rec_mesh=rec_mesh.crop(bbox)

## sample mesh for distance calculation
scanned_points_rec = rec_mesh.sample_points_uniformly(number_of_points=len(scanned_points.points))
visualize_pcd([scanned_points_rec])
##### get projection of each z height
for z in np.arange(4,z_max+resolution_z,resolution_z):
    print(z)
    #### crop z height
    min_bound = (-1e5,-1e5,z-windows_z/2)
    max_bound = (1e5,1e5,z+windows_z/2)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
    points_proj=scanned_points_rec.crop(bbox)
    visualize_pcd([points_proj])
    ##################

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
        all_welds[weld_i][z]={}
        y_min=np.min(np.asarray(welds_points.points)[:,1])
        y_max=np.max(np.asarray(welds_points.points)[:,1])
        for y in np.arange(y_min,y_max+resolution_y,resolution_y):
            min_bound = (-1e5,y-windows_y/2,-1e5)
            max_bound = (1e5,y+windows_y/2,1e5)
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
            welds_points_y = welds_points.crop(bbox)
            if len(welds_points_y.points)<stop_thres_w:
                # all_welds[weld_i][z][y]=0
                continue
            ### get the width
            sort_x_value=np.sort(np.asarray(welds_points_y.points)[:,0])
            x_min = np.mean(sort_x_value[:use_points_num])
            x_max = np.mean(sort_x_value[-use_points_num:])
            this_width=x_max-x_min
            all_welds[weld_i][z][y]=this_width
            # visualize_pcd([welds_points_y])
        ### get all zy coord
        y_coord=np.array(list(all_welds[weld_i][z].keys()))
        y_width=np.array(list(all_welds[weld_i][z].values()))
        plt.plot(y_coord,y_width,marker='o',color=table_colors[weld_i],label='Weld Number '+str(weld_i))
        welds_points.paint_uniform_color(mcolors.to_rgb(table_colors[weld_i]))
        all_welds_points+=welds_points

    # points_proj.paint_uniform_color([1, 0, 0])
    # welds_points.paint_uniform_color([0, 0.8, 0])
    visualize_pcd([all_welds_points])
    plt.show()
    # exit()