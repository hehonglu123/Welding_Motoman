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

###MTI connect to RR
mti_client = RRN.ConnectService("rr+tcp://192.168.55.10:60830/?service=MTI2D")
mti_client.setExposureTime("25")

while True:

    scan_points=np.array([mti_client.lineProfile.X_data,np.zeros(len(mti_client.lineProfile.Z_data)),mti_client.lineProfile.Z_data,])
    scan_points[0]=scan_points[0]*-1 # reversed x-axis
    scan_points=scan_points.T

    pcd = o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(scan_points)

    d=70
    xaxis=0

    width=0.1
    height=0.1
    bbox_mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=d)
    box_move=np.eye(4)
    box_move[0,3]=-width/2+xaxis # x-axis
    box_move[1,3]=-height/2 # y-axis
    box_move[2,3]=0
    bbox_mesh.transform(box_move)

    visualize_pcd([pcd,bbox_mesh])

