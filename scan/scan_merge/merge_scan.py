import sys
sys.path.append('../toolbox/')
from robot_def import *

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

robot=robot_obj('MA_1440_A0',def_path='../config/MA_1440_A0_robot_default_config.yml',tool_file_path='../config/scanner_tcp2.csv',\
	pulse2deg_file_path='../config/MA_1440_A0_pulse2deg.csv')
q1=np.radians([31.1143,57.8258,10.2037,-0.4206,35.1934,-29.7721])
q2=np.radians([38.7598,59.7069,18.4387,-45.9732,34.4456,29.4706])
q3=np.radians([47.5995,59.7092,18.4380,-45.9742,46.3366,57.3870])

pose1=robot.fwd(q1)
pose2=robot.fwd(q2)
pose3=robot.fwd(q3)

points1 = np.loadtxt('../points1.csv',delimiter=",", dtype=np.float64)
points2 = np.loadtxt('../points2.csv',delimiter=",", dtype=np.float64)
points3 = np.loadtxt('../points3.csv',delimiter=",", dtype=np.float64)

points1=np.dot(pose1.R,points1[:,:3].T).T+np.tile(pose1.p,(len(points1),1))
points2=np.dot(pose2.R,points2[:,:3].T).T+np.tile(pose2.p,(len(points2),1))
points3=np.dot(pose3.R,points3[:,:3].T).T+np.tile(pose3.p,(len(points3),1))

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(points1[:,0],points1[:,1],points1[:,2],c='red',s=1,label='scan 1')
ax.scatter(points2[:,0],points2[:,1],points2[:,2],c='green',s=1,label='scan 2')
ax.scatter(points3[:,0],points3[:,1],points3[:,2],c='blue',s=1,label='scan 3')
plt.show()