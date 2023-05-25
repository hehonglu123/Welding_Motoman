import sys
import matplotlib
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

data_dir='../../data/wall_weld_test/scan_cont_newdx_1/scans/'

scan_points = np.load(data_dir + 'points_'+str(0)+'.npy')
pcd = o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(scan_points)

d=243
width=0.5
height=0.5
bbox_mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=d)
box_move=np.eye(4)
box_move[0,3]=-width/2 # x-axis
box_move[1,3]=-height/2 # y-axis
box_move[2,3]=-d
bbox_mesh.transform(box_move)

visualize_pcd([pcd,bbox_mesh])

