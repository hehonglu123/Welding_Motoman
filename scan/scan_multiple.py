from RobotRaconteur.Client import *
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

client=RRN.ConnectService('rr+tcp://192.168.55.27:64238?service=scanner')

scan_points_list=[]
scan_iteration=10

scan_st=time.perf_counter()
for i in range(scan_iteration):
    st=time.perf_counter()
    mesh=client.capture(True)
    scan_points = RRN.NamedArrayToArray(mesh.vertices)
    print("Points Num:",len(scan_points))
    scan_points_list.append(scan_points)
    print("Frame Elapes:",time.perf_counter()-st)
scan_et=time.perf_counter()
print("FPS:",scan_iteration/(scan_et-scan_st))

ax = plt.figure().add_subplot(projection='3d')
for i in range(scan_iteration):
    ax.scatter(scan_points_list[i][:,0],scan_points_list[i][:,1],scan_points_list[i][:,2],s=1)

    ## save points
    np.savetxt('multi_scan_test/points_'+str(i)+'.csv',scan_points_list[i],delimiter=',')
plt.show()