import sys
import matplotlib

from sklearn import cluster
sys.path.append('../../toolbox/')
sys.path.append('../scan_tools/')
from robot_def import *
from scan_utils import *
from general_robotics_toolbox import *
import open3d as o3d

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
from copy import deepcopy
import colorsys
import math

scanned_points_mesh = o3d.io.read_triangle_mesh('../jw_example/deferred_captured_mesh/deferred_captured_mesh_1.stl')
print(len(scanned_points_mesh.vertices))
visualize_pcd([scanned_points_mesh])

pcd_combined = o3d.geometry.PointCloud()
pcd_combined.points=scanned_points_mesh.vertices
visualize_pcd([scanned_points_mesh,pcd_combined])

exit()

data_dir='../../data/wall_weld_test/scan_cont_1/'
config_dir='../../config/'

pcd_combined = o3d.geometry.PointCloud()
for i in range(len(100)):
    points = np.save(data_dir + 'points_'+str(i)+'.npy',)

    pcd = o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=0.1)

    pcd_combined += pcd
visualize_pcd([pcd_combined])