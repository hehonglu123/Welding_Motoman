import sys

from sklearn import cluster
sys.path.append('../../toolbox/')
from robot_def import *
from general_robotics_toolbox import *
import open3d as o3d

import numpy as np
import cv2
import matplotlib.pyplot as plt
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

def visualize_pcd(show_pcd_list):
	points_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3,origin=[0,0,0])
	show_pcd_list.append(points_frame)
	o3d.visualization.draw_geometries(show_pcd_list,width=960,height=540)
	# o3d.visualization.draw(show_pcd_list,width=960,height=540)

data_dir='test2/'
config_dir='../../config/'

scan_resolution=5 #scan every 5 mm
scan_per_pose=3 # take 3 scan every pose

robot=robot_obj('MA_1440_A0',def_path=config_dir+'MA_1440_A0_robot_default_config.yml',tool_file_path=config_dir+'scanner_tcp2.csv',\
	pulse2deg_file_path=config_dir+'MA_1440_A0_pulse2deg.csv')

cart_p=[]
joints_p=np.loadtxt(data_dir+'scan_js.csv',delimiter=",", dtype=np.float64)
for i in range(len(joints_p)):
	joints_p[i][5] = -joints_p[i][5]
	joints_p[i][3] = -joints_p[i][3]
	joints_p[i][1] = 90 - joints_p[i][1]
	joints_p[i][2] = joints_p[i][2] + joints_p[i][1]

joints_p=np.radians(joints_p)
for q in joints_p:
	cart_p.append(robot.fwd(q))
print("Joint Space")
print(np.degrees(joints_p))
print("Cart Space")
print(cart_p)

curve=[]
curve_R=[]
curve_js=[]
total_step=0
for i in range(len(cart_p)-1):
	travel_vec=cart_p[i+1].p-cart_p[i].p	
	travel_dis=np.linalg.norm(travel_vec)
	travel_vec=travel_vec/travel_dis*scan_resolution
	print("Travel Vector:",travel_vec)
	print("Travel Distance:",travel_dis)

	xp=np.append(np.arange(cart_p[i].p[0],cart_p[i+1].p[0],travel_vec[0]),cart_p[i+1].p[0])
	yp=np.append(np.arange(cart_p[i].p[1],cart_p[i+1].p[1],travel_vec[1]),cart_p[i+1].p[1])
	zp=np.append(np.arange(cart_p[i].p[2],cart_p[i+1].p[2],travel_vec[2]),cart_p[i+1].p[2])
	print(len(xp))
	print(len(yp))
	print(len(zp))

	for travel_i in range(len(xp)):
		this_p=Transform(cart_p[i].R,[xp[travel_i],yp[travel_i],zp[travel_i]])
		curve.append(this_p.p)
		curve_R.append(this_p.R)
		if len(curve_js)!=0:
			curve_js.append(robot.inv(this_p.p,this_p.R,curve_js[-1])[0])
		else:
			curve_js.append(robot.inv(this_p.p,this_p.R,joints_p[0])[0])
	total_step+=len(xp)
curve=np.array(curve)
curve_js=np.array(curve_js)
print(curve)
# print(np.degrees(curve_js))
print("Total step:",total_step)

T_base_frame1 = Transform(curve_R[0],curve[0])

## move bricks to origin
T_origin_R=rot([0,0,1],np.radians(87.5))
T_origin=Transform(T_origin_R,np.dot(T_origin_R,-curve[0])+np.array([-3,-19.1,243.9]))
print(T_origin)

#### processing parameters
voxel_size=0.1

pcd_combined = o3d.geometry.PointCloud()
for i in range(len(curve)):
# for i in range(5):
	for scan_i in range(scan_per_pose):
		points = np.loadtxt(data_dir + 'points_'+str(i)+'_'+str(scan_i)+'.csv',delimiter=",", dtype=np.float64)
		
		points = np.transpose(np.matmul(curve_R[i],np.transpose(points)))+curve[i]
		## get the points closed to origin
		points = np.transpose(np.matmul(T_origin.R,np.transpose(points)))+T_origin.p

		###### preprocessing
		## to pcd
		pcd = o3d.geometry.PointCloud()
		pcd.points=o3d.utility.Vector3dVector(points)
		# visualize_pcd([pcd])
		# exit()
		## voxel down sample
		pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
		## add to combined pcd
		pcd_combined += pcd

#### processing
# visualize_pcd([pcd_combined])
## voxel down sample
pcd_combined = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
# visualize_pcd([pcd_combined])
# exit()
## crop point clouds
min_bound = (-1,-1,-1)
max_bound = (143.1+5,15.8+1,30.6+1)
bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
pcd_combined=pcd_combined.crop(bbox)
# visualize_pcd([pcd_combined])
## outlier removal
nb_neighbors=40
std_ratio=0.5
cl,ind=pcd_combined.remove_statistical_outlier(nb_neighbors=nb_neighbors,std_ratio=std_ratio)
# display_inlier_outlier(pcd_combined,ind)
# exit()
pcd_combined=cl
## DBSCAN pcd clustering
cluster_neighbor=0.75
min_points=50
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd_combined.cluster_dbscan(eps=cluster_neighbor, min_points=min_points, print_progress=True))
max_label=labels.max()
print("Cluster count:",labels.max()+1)
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
# pcd_combined.colors = o3d.utility.Vector3dVector(colors[:, :3])
# visualize_pcd([pcd_combined])
pcd_combined=pcd_combined.select_by_index(np.argwhere(labels>=0))
## plane reconstruction
# pcd_combined.estimate_normals(
#     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
#         pcd_combined, depth=9)
# pcd_combined.points=mesh.vertices

# visualize_pcd([pcd_combined])

######## compare with scan mesh
scan_mesh=o3d.io.read_triangle_mesh(data_dir+'../lego_brick/scan_mesh.stl')
scan_mesh.compute_vertex_normals()
# visualize_pcd([scan_mesh])
scan_mesh_points = o3d.io.read_point_cloud(data_dir+'../lego_brick/scan_mesh_points.pcd')
## register points (dont need this after the motion trackers (?))
min_bound = (-1,-1,19.3)
max_bound = (143.1+5,15.8+1,30.6+1)
bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
scan_mesh_points=scan_mesh_points.crop(bbox)
# visualize_pcd([scan_mesh_points])
pcd_combined_reg = pcd_combined.voxel_down_sample(voxel_size=0.5)
# visualize_pcd([pcd_combined_reg])
trans_init=np.eye(4)
threshold = 1
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd_combined_reg, scan_mesh_points, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
print(reg_p2p)
print(reg_p2p.transformation)
visualize_pcd([pcd_combined,scan_mesh])
pcd_combined_reg.transform(reg_p2p.transformation)
pcd_combined.transform(reg_p2p.transformation)
pcd_combined_reg.paint_uniform_color([1, 0.706, 0])
scan_mesh_points.paint_uniform_color([0, 0.651, 0.929])
# visualize_pcd([pcd_combined_reg,scan_mesh_points])
visualize_pcd([pcd_combined,scan_mesh])
##

## crop scan mesh surfaces
normals = np.asarray(scan_mesh.triangle_normals)
normal_z_id = np.squeeze(np.argwhere(np.abs(normals[:,2])==1))
scan_mesh_target=deepcopy(scan_mesh)
scan_mesh_target.triangles = o3d.utility.Vector3iVector(
    np.asarray(scan_mesh.triangles)[normal_z_id, :])
scan_mesh_target.triangle_normals = o3d.utility.Vector3dVector(
    np.asarray(scan_mesh.triangle_normals)[normal_z_id, :])
visualize_pcd([scan_mesh_target])
visualize_pcd([scan_mesh_target,pcd_combined])
## calculate distance to mesh
scan_mesh_target_t = o3d.t.geometry.TriangleMesh.from_legacy(scan_mesh_target)
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(scan_mesh_target_t)
query_points = o3d.core.Tensor(np.asarray(pcd_combined.points), dtype=o3d.core.Dtype.Float32)
unsigned_distance = scene.compute_distance(query_points)
unsigned_distance = unsigned_distance.numpy()
min_dist=np.min(unsigned_distance)
max_dist=np.max(unsigned_distance)
print("unsigned distance", unsigned_distance)
print("unsigned distance min", min_dist)
print("unsigned distance min", max_dist)
low_bound=0
high_bound=max_dist+0.1
color_dist = plt.get_cmap("rainbow")((unsigned_distance-low_bound)/(high_bound-low_bound))
pcd_combined_dist_scan=deepcopy(pcd_combined)
pcd_combined_dist_scan.colors = o3d.utility.Vector3dVector(color_dist[:, :3])
visualize_pcd([scan_mesh_target,pcd_combined_dist_scan])

######## compare with target mesh
target_mesh=o3d.io.read_triangle_mesh(data_dir+'../lego_brick/target_mesh.stl')
target_mesh.compute_vertex_normals()
min_bound = np.array([0,0,19.2])
max_bound = np.array([143.1,15.8,21])
margin=0.01
min_bound=min_bound-margin
max_bound=max_bound+margin
bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
target_mesh_target=target_mesh.crop(bbox)
visualize_pcd([target_mesh_target,pcd_combined])

## calculate distance to mesh
target_mesh_target_t = o3d.t.geometry.TriangleMesh.from_legacy(target_mesh_target)
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(target_mesh_target_t)
query_points = o3d.core.Tensor(np.asarray(pcd_combined.points), dtype=o3d.core.Dtype.Float32)
unsigned_distance = scene.compute_distance(query_points)
unsigned_distance = unsigned_distance.numpy()
min_dist=np.min(unsigned_distance)
max_dist=np.max(unsigned_distance)
print("unsigned distance", unsigned_distance)
print("unsigned distance min", min_dist)
print("unsigned distance min", max_dist)
low_bound=0
high_bound=max_dist+0.1
color_dist = plt.get_cmap("rainbow")((unsigned_distance-low_bound)/(high_bound-low_bound))
pcd_combined_dist_target=deepcopy(pcd_combined)
pcd_combined_dist_target.colors = o3d.utility.Vector3dVector(color_dist[:, :3])
visualize_pcd([target_mesh_target,pcd_combined_dist_target])