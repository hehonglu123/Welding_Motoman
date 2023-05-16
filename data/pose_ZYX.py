
import numpy as np
import sys
from general_robotics_toolbox import *
sys.path.append('../toolbox/')
from utils import *
from robot_def import *



dataset='blade0.1/'
solution_dir='NX_slice2/'
data_dir=dataset+solution_dir

###reference frame transformation
curve_pose=np.loadtxt(data_dir+'curve_pose.csv',delimiter=',')

tool=np.loadtxt('../config/torch.csv',delimiter=',')

print(tool[:-1,-1])
print(np.degrees(rotationMatrixToEulerAngles(tool[:3,:3])))

