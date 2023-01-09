import sys
sys.path.append('../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *
from MotionSendMotoman import *

from RobotRaconteur.Client import *
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

ms=MotionSend()

points1=np.loadtxt('points1.csv',delimiter=',')
points2=np.loadtxt('points2.csv',delimiter=',')
points3=np.loadtxt('points3.csv',delimiter=',')


points1=np.dot(pose1.R,points1[:,:3].T).T+np.tile(pose1.p,(len(points1),1))
points2=np.dot(pose2.R,points2[:,:3].T).T+np.tile(pose2.p,(len(points2),1))
points3=np.dot(pose3.R,points3[:,:3].T).T+np.tile(pose3.p,(len(points3),1))

points=np.vstack((points1,points2,points3))
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2])


plt.show()