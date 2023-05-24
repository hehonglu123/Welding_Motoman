import sys
import matplotlib
sys.path.append('../../toolbox/')
sys.path.append('../scan_tools/')
from robot_def import *
from scan_utils import *
from scan_continuous import *
from general_robotics_toolbox import *
import open3d as o3d
from RobotRaconteur.Client import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import time
from copy import deepcopy
import colorsys
import math

c = RRN.ConnectService('rr+tcp://localhost:64238?service=scanner')
cscanner = ContinuousScanner(c)
cscanner.start_capture()
time.sleep(0.5)
cscanner.end_capture()
scans,scan_stamps=cscanner.get_capture()
for scan in scans:
    scan_points = RRN.NamedArrayToArray(scan.vertices)
    break # take the first scan

pcd = o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(scan_points)

d=245
width=0.5
height=0.5
bbox_mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=d)
box_move=np.eye(4)
box_move[0,3]=-width/2 # x-axis
box_move[1,3]=-height/2 # y-axis
box_move[2,3]=-d
bbox_mesh.transform(box_move)

visualize_pcd([pcd,bbox_mesh])

