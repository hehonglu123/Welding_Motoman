from RobotRaconteur.Client import *
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

client=RRN.ConnectService('rr+tcp://192.168.55.27:64238?service=scanner')

mesh=client.capture(True)
scan_points = RRN.NamedArrayToArray(mesh.vertices)
print("Points Num:",len(scan_points))

ax = plt.figure().add_subplot(projection='3d')
ax.scatter(scan_points[:,0],scan_points[:,1],scan_points[:,2])
plt.show()