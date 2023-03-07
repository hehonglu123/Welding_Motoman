from general_robotics_toolbox import *
import open3d as o3d
import numpy as np
from copy import deepcopy

import sys
sys.path.append('../../../toolbox/')
sys.path.append('../../../scan/scan_tools/')
from scan_utils import *

artec_mesh=o3d.io.read_triangle_mesh('meshes/artec_spider.stl')
artec_mesh.compute_vertex_normals()

visualize_pcd([artec_mesh])
trans=[]
for i in range(6):
    trans.append(float(input("x,y,z,r,p,y:")))

while True:
    trans=np.array(trans)
    print(trans)
    print(trans[3:])
    print(trans[:3])
    trans_T=np.hstack((rpy2R(np.radians(trans[3:])),np.array([trans[:3]]).T))
    trans_T=np.vstack((trans_T,[0,0,0,1]))
    artec_mesh.transform(trans_T)
    visualize_pcd([artec_mesh])

    trans=[]
    for i in range(6):
        a=input("x,y,z,r,p,y:")
        if a=='q':
            o3d.io.write_triangle_mesh('meshes/artec_mount.stl',artec_mesh)
            exit()
        trans.append(float(a))

    # artec_mesh.transform(np.linalg.inv(trans_T))

