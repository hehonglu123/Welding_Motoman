import sys, copy
import matplotlib

from sklearn import cluster
sys.path.append('../toolbox/')
from pointcloud_toolbox import *
from utils import *
from robot_def import *
# from scan_utils import *
from general_robotics_toolbox import *
import open3d as o3d

import numpy as np

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

    
data_dir='../data/blade0.1/'
scanned_dir='../../evaluation/Blade_ER70S6/'
######## read the scanned stl
target_mesh = o3d.io.read_triangle_mesh(data_dir+'surface.stl')
scanned_mesh = o3d.io.read_triangle_mesh(scanned_dir+'no_base_layer.stl')
target_mesh.compute_vertex_normals()
scanned_mesh.compute_vertex_normals()

## inch to mm
target_mesh.scale(25.4, center=(0, 0, 0))
# visualize_pcd([target_mesh,scanned_mesh])

## sample as pointclouds
target_points = target_mesh.sample_points_uniformly(number_of_points=111000)
scanned_points = scanned_mesh.sample_points_uniformly(number_of_points=111000)

## paint colors
target_points = target_points.paint_uniform_color([0, 0.8, 0.0])
scanned_points = scanned_points.paint_uniform_color([0.8, 0, 0.0])
scanned_points_original=copy.deepcopy(scanned_points)
total_transformation = np.eye(4)
## global tranformation
R_guess,p_guess=global_alignment(scanned_points.points,target_points.points)
total_transformation = H_from_RT(R_guess,p_guess)
scanned_points=scanned_points.transform(total_transformation)


visualize_pcd([target_points,scanned_points])




# ## ICP
# icp_turns = 1
# threshold=5
# max_iteration=1000
# for i in range(icp_turns):
#     reg_p2p = o3d.pipelines.registration.registration_icp(
#                 scanned_points, target_points, threshold, np.eye(4),
#                 o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#                 o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
#     scanned_points=scanned_points.transform(reg_p2p.transformation)
#     total_transformation = reg_p2p.transformation@total_transformation

# print("Final Transformation:",total_transformation)
# print("Fitness:",reg_p2p.fitness)
# print("inlier_rmse:",reg_p2p.inlier_rmse)
# print(reg_p2p)
# visualize_pcd([target_points,scanned_points])
