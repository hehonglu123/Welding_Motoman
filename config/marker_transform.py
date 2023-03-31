from general_robotics_toolbox import *
import numpy as np

R = rot([0,0,1],np.pi/2)
R = R@rot([1,0,0],np.pi/2)
T = Transform(R,[423.9483,-49.9776,0])
T = T.inv()

marker_p = [np.array([-26.8060,70.7568,-38.8455]),np.array([153.1176,-46.1570,67.5])\
            ,np.array([126.4099,-124.3,7.8978]),np.array([153.1176,-46.1570,-67.5])]
        
for i in range(len(marker_p)):
    print(np.matmul(T.R,marker_p[i])+T.p)