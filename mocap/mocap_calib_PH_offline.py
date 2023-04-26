from copy import deepcopy
import sys
sys.path.append('../toolbox/')
sys.path.append('../redundancy_resolution/')
from utils import *
from robot_def import * 

import numpy as np
import time
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

config_dir='../config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',\
pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

base_marker_config_file=config_dir+'MA2010_marker_config.yaml'
with open(base_marker_config_file,'r') as file:
    base_marker_data = yaml.safe_load(file)

H = []
H_point = []
for i in range(6):
    H.append(np.array([base_marker_data['H'][i]['x'],
                       base_marker_data['H'][i]['y'],
                       base_marker_data['H'][i]['z']]))
    H_point.append(np.array([base_marker_data['H_point'][i]['x'],
                       base_marker_data['H_point'][i]['y'],
                       base_marker_data['H_point'][i]['z']]))
H=np.array(H).T
H_point=np.array(H_point).T

print(H)

for i in range(6):
    H[:,i]=H[:,i]/np.linalg.norm(H[:,i])

# rotate R
z_axis = H[:,0]
y_axis = H[:,1]
y_axis = y_axis-np.dot(z_axis,y_axis)*z_axis
y_axis = y_axis/np.linalg.norm(y_axis)
x_axis = np.cross(y_axis,z_axis)
R = np.array([x_axis,y_axis,z_axis])

H = np.matmul(R,H)
print(H.T)
H_point = (H_point.T-H_point[:,0]).T
H_point = np.matmul(R,H_point)
print(H_point.T)

diff_H = np.linalg.norm(robot_weld.robot.H-H,axis=0)
print(diff_H)

for i in range(6):
    print(np.degrees(np.arccos(np.dot(H[:,i],robot_weld.robot.H[:,i])/(np.linalg.norm(H[:,i])*np.linalg.norm(robot_weld.robot.H[:,i])))))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for i in range(6):
    start_p = H_point[:,i]-H[:,i]*1000
    end_p = H_point[:,i]+H[:,i]*1000
    ax.plot([start_p[0],end_p[0]], [start_p[1],end_p[1]], [start_p[2],end_p[2]], label='axis '+str(i+1))
plt.legend()
plt.show()