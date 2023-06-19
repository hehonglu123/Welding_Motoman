from general_robotics_toolbox import *
import numpy as np

# weldgun markers
R = np.array([[0.,-1,0],[0,0,-1],[1,0,0]]).T
R = R@rot([0,1,0],np.radians(22))
print(R)
T = Transform(R,[423.9483,-49.9776,0])
T = T.inv()
marker_p = [np.array([-26.8060,71.6229,-39.3455]),np.array([152.1240,-46.1204,68.5])\
            ,np.array([126.4099,-125.3,7.8978]),np.array([152.1240,-46.1204,-68.5])]

# scanner markers
# R = np.array([[-1,0,0],[0,0.954709,0.297542],[0,0.297542,-0.954709]])
# R[:,1] = R[:,1]/np.linalg.norm(R[:,1])
# R[:,2] = R[:,2]/np.linalg.norm(R[:,2])
# T_camera_bottom = Transform(R,[0,120.97,16.053]) # T^camera_bottom
# R = np.array([[-1,0,0],[0,0,1],[0,1,0]]).T
# T_bottom_cad = Transform(R,[0.,0.,0.])
# T_camera_cad = T_bottom_cad*T_camera_bottom
# T = T_camera_cad.inv() # T^cad_camera
# marker_p = [np.array([0,-173.2886,28.0355]),np.array([93.1331,-74.9856,20.6397])\
#             ,np.array([0.4901,-72.1599,208.3419]),np.array([-93.1331,-74.9856,20.6397])]
        
for i in range(len(marker_p)):
    print(np.matmul(T.R,marker_p[i])+T.p)