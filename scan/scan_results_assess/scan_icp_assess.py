import sys
import matplotlib

from sklearn import cluster
sys.path.append('../../toolbox/')
sys.path.append('../scan_tools')
from utils import *
from robot_def import *
from scan_utils import *
from general_robotics_toolbox import *
import open3d as o3d

import numpy as np

data_dir='../data/'

######## read the scanned stl
target_mesh = o3d.io.read_triangle_mesh(data_dir+'surface.stl')
scanned_mesh = o3d.io.read_triangle_mesh(data_dir+'no_base_layer.stl')
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

total_transformation = np.eye(4)
## global tranformation
## ICP is not good at global transformation.
## Provide a roughly guess transformation.
global_T_guess = Transform(rot(np.array([0,0,1]),-np.pi/2)@rot(np.array([0,1,0]),np.radians(180)),[100,0,-25])
scanned_points=scanned_points.transform(H_from_RT(global_T_guess.R,global_T_guess.p))
total_transformation = total_transformation@H_from_RT(global_T_guess.R,global_T_guess.p)
# visualize_pcd([target_points,scanned_points])

## ICP
icp_turns = 1
threshold=5
max_iteration=1000
for i in range(icp_turns):
    reg_p2p = o3d.pipelines.registration.registration_icp(
                scanned_points, target_points, threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    scanned_points=scanned_points.transform(reg_p2p.transformation)
    total_transformation = reg_p2p.transformation@total_transformation

print("Final Transformation:",total_transformation)
print("Fitness:",reg_p2p.fitness)
print("inlier_rmse:",reg_p2p.inlier_rmse)
print(reg_p2p)
visualize_pcd([target_points,scanned_points])

