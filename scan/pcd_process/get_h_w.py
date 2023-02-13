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
from matplotlib import colors
import time
from copy import deepcopy
import colorsys
import math

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

##### get projection of each z height
z_max=np.max(np.asarray(scanned_points.points)[:,2])
resolution=0.1
for z in np.arange(4,z_max+resolution,resolution):
    print(z)
    min_bound = (-1e5,-1e5,z-resolution/2)
    max_bound = (1e5,1e5,z+resolution/2)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
    points_proj=scanned_points.crop(bbox)
    visualize_pcd([points_proj,y_axis_mesh])


####### estimate normal
# scanned_points.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# visualize_pcd([scanned_points],point_show_normal=True)
# ####### estimate mesh
# alpha = 0.03
# rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(scanned_points, alpha)
# rec_mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([rec_mesh])
# radii = [0.005, 0.01, 0.02, 0.04]
# rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     scanned_points, o3d.utility.DoubleVector(radii))
# o3d.visualization.draw_geometries([scanned_points, rec_mesh])
# with o3d.utility.VerbosityContextManager(
#         o3d.utility.VerbosityLevel.Debug) as cm:
#     rec_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
#         scanned_points, depth=9)
# densities = np.asarray(densities)
# density_colors = plt.get_cmap('plasma')(
#     (densities - densities.min()) / (densities.max() - densities.min()))
# density_colors = density_colors[:, :3]
# density_mesh = o3d.geometry.TriangleMesh()
# density_mesh.vertices = rec_mesh.vertices
# density_mesh.triangles = rec_mesh.triangles
# density_mesh.triangle_normals = rec_mesh.triangle_normals
# density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
# vertices_to_remove = densities < np.quantile(densities, 0.01)
# density_mesh.remove_vertices_by_mask(vertices_to_remove)
# o3d.visualization.draw_geometries([density_mesh])