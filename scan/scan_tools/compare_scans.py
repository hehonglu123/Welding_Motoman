from RobotRaconteur.Client import *
import threading
import time
from contextlib import suppress
import numpy as np
import open3d as o3d
from scan_utils import *

c = RRN.ConnectService('rr+tcp://192.168.55.27:64238?service=scanner')
print("here")
time.sleep(8)

mesh=c.capture(True)
scan_single_points = RRN.NamedArrayToArray(mesh.vertices)

mesh_handles=c.capture_deferred(False)
prepare_gen = c.deferred_capture_prepare_stl(mesh_handles)
with suppress(RR.StopIterationException):
    prepare_res = prepare_gen.Next()
stl_mesh = c.getf_deferred_capture(mesh_handles)
scan_cont_points = RRN.NamedArrayToArray(stl_mesh.vertices)

pcd_single = o3d.geometry.PointCloud()
pcd_single.points=o3d.utility.Vector3dVector(np.array(scan_single_points))
pcd_single.paint_uniform_color([1, 0, 0])

pcd_cont = o3d.geometry.PointCloud()
pcd_cont.points=o3d.utility.Vector3dVector(np.array(scan_cont_points))
pcd_cont.paint_uniform_color([0, 1, 0])

print("Single Scan points N:",len(scan_single_points))
print("Continuous Scan points N:",len(scan_cont_points))

# visualize_pcd([pcd_cont,pcd_single])
np.savetxt('data/points_single.csv',np.array(scan_single_points),delimiter=',')
np.savetxt('data/points_cont.csv',np.array(scan_cont_points),delimiter=',')