import numpy as np
from general_robotics_toolbox import *

position_x = -86.361
position_y = -39.720
position_z = 585.557

orientation_Rx = -155.4050
orientation_Ry = -29.4499
orientation_Rz = -100.2860

outputToolName = 'fujicam'

H = np.eye(4)
H[0:3,3] = np.array([position_x, position_y, position_z])
H[0:3,0:3] = rpy2R(np.radians([orientation_Rx, orientation_Ry, orientation_Rz]))

np.savetxt(outputToolName+'.csv', H, delimiter=',')