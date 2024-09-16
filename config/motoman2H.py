import numpy as np
import sys
sys.path.append('../toolbox/')
from robot_def import *

center2bottom=380
# x=1652.727
# y=-814.148
# z=-432.194
# rx=-0.0733
# ry=14.9492
# rz=90.2564

x=1652.559
y=-815.108
z=-432.247
rx=0.3187
ry=16.2177
rz=90.5345

R=Rz(np.radians(-rz))@Ry(np.radians(-ry))@Rx(np.radians(-rx))
H=H_from_RT(R,[x,y-center2bottom*np.sin(np.radians(ry)),z-6-center2bottom*np.cos(np.radians(ry))])
print(H)