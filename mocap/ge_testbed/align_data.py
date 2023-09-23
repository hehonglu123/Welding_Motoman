import numpy as np
from copy import deepcopy
from general_robotics_toolbox import *
import pickle
import time
import sys
sys.path.append('../../toolbox/')
from robot_def import *
from matplotlib import pyplot as plt

from qpsolvers import solve_qp

from numpy.random import default_rng
rng = default_rng()

Rx=np.array([1,0,0])
Ry=np.array([0,1,0])
Rz=np.array([0,0,1])

robot_type='R1'
dataset_date='0913'
datasets='test'+dataset_date+'_'+robot_type+'_part1/train_data'

if robot_type=='R1':
    config_dir='../../config/'
    robot_name='M10ia'
    tool_name='ge_R1_tool'
    robot_marker_dir=config_dir+robot_name+'_marker_config/'
    tool_marker_dir=config_dir+tool_name+'_marker_config/'
    uframe_num=4
    utool_num=1
    robot_group=1

    # only R matter
    nominal_robot_base = Transform(np.array([[0,-1,0],
                                        [0,0,1],
                                        [-1,0,0]]),[0,0,0]) 

elif robot_type=='R2':
    config_dir='../../config/'
    robot_name='LRMATE200id'
    tool_name='ge_R2_tool'
    robot_marker_dir=config_dir+robot_name+'_marker_config/'
    tool_marker_dir=config_dir+tool_name+'_marker_config/'
    uframe_num=4
    utool_num=1
    robot_group=2
    
    # only R matter
    nominal_robot_base = Transform(np.array([[0,1,0],
                                        [0,0,1],
                                        [1,0,0]]),[0,0,0]) 

print("Dataset Date:",dataset_date)

robot=robot_obj(robot_name,def_path=config_dir+robot_name+'_robot_default_config.yml',tool_file_path=config_dir+tool_name+'.csv',\
base_marker_config_file=robot_marker_dir+robot_name+'_marker_config.yaml',tool_marker_config_file=tool_marker_dir+tool_name+'_marker_config.yaml')

raw_data_dir='PH_grad_data/'+datasets

robot_q =np.loadtxt(raw_data_dir+'_robot_q_raw.csv',delimiter=',')
print(len(robot_q))
print(len(robot_q)/3)

with open(raw_data_dir+'_marker_raw.pickle', 'rb') as handle:
    marker_T=pickle.load(handle)

cond_thres=4
split_thres=4
for mid in marker_T.keys():
    
    last_pose=deepcopy(marker_T[mid][0])
    dist=[]
    for pose in marker_T[mid]:
        if pose[-1]<cond_thres:
            continue
        
        dist.append(np.linalg.norm(pose[:3]-last_pose[:3]))
        last_pose=deepcopy(pose)
        
    plt.plot(dist,'-o')
    plt.show()
    