import numpy as np
import sys
# sys.path.append('../toolbox/')
# from robot_def import *
from motoman_def import *

center2bottom=380
# x=1652.727
# y=-814.148
# z=-432.194
# rx=-0.0733
# ry=14.9492
# rz=90.2564

# x=1652.559
# y=-815.108
# z=-432.247
# rx=0.3187
# ry=16.2177
# rz=90.5345

x=1652.831
y=-813.800
z=-433.622
rx=0.0028
ry=14.6701
rz=90.2315

R=Rz(np.radians(rz))@Ry(np.radians(ry))@Rx(np.radians(rx))
H=H_from_RT(R,[x,y-center2bottom*np.sin(np.radians(ry)),z-6-center2bottom*np.cos(np.radians(ry))])
print("positioner",H)

# np.savetxt('D500B_pose.csv',H,delimiter=',')

x=-59.417
y=-1.737
z=581.741
rx=-177.5269
ry=-36.8119
rz=-8.4629

R=Rz(np.radians(rz))@Ry(np.radians(ry))@Rx(np.radians(rx))
H=H_from_RT(R,[x,y,z])
print("tool",H)
np.savetxt('fujicam.csv',H,delimiter=',')