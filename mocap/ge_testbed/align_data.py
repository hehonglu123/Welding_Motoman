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
datasets='test'+dataset_date+'_'+robot_type+'_part2/train_data'

if robot_type=='R1':
    config_dir='config/'
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
    config_dir='config/'
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
base_marker_config_file=robot_marker_dir+robot_name+'_'+dataset_date+'_marker_config.yaml',tool_marker_config_file=tool_marker_dir+tool_name+'_'+dataset_date+'_marker_config.yaml')

raw_data_dir='PH_grad_data/'+datasets

robot_q =np.loadtxt(raw_data_dir+'_robot_q_raw.csv',delimiter=',')

robot_q_align = []
for q_id in range(0,len(robot_q),3):
    this_q = np.mean(robot_q[q_id:q_id+3],axis=0)
    robot_q_align.append(this_q)
robot_q_align=np.array(robot_q_align)
if robot_type=='R1':
    robot_q_align=robot_q_align[:,:6]
elif robot_type=='R2':
    robot_q_align=robot_q_align[:,6:12]

# exit()

# with open(raw_data_dir+'_marker_raw.pickle', 'rb') as handle:
#     marker_T=pickle.load(handle)

# cond_thres=4
# split_thres=5
# for mid in marker_T.keys():
#     print("Marker id:",mid)
    
#     last_pose=deepcopy(marker_T[mid][0])
#     dist=[]
#     for pose in marker_T[mid]:
#         if pose[-1]>cond_thres:
#             continue
        
#         dist.append(np.linalg.norm(pose[:3]-last_pose[:3]))
#         last_pose=deepcopy(pose)
    
#     dist=np.array(dist)
#     print(np.count_nonzero(dist>split_thres))
    
#     plt.plot(dist,'-o')
#     plt.show()
    
tool_T =np.loadtxt(raw_data_dir+'_tool_T_raw.csv',delimiter=',')

cond_thres=-1
split_thres=5

cut_start=[173700,221940]
cut_end=[173860,222120]
del_id = []
for cut_i in range(len(cut_start)):
    del_id = np.append(del_id,np.arange(cut_start[cut_i],cut_end[cut_i]))
del_id=del_id.astype(int)

tool_T = np.delete(tool_T,del_id,axis=0)

last_pose=deepcopy(tool_T[0])
dist=[]
for pose in tool_T:
    if pose[-1]<cond_thres:
        continue
    
    dist.append(np.linalg.norm(pose[:3]-last_pose[:3]))
    last_pose=deepcopy(pose)

dist=np.array(dist)
split_id = np.where(dist>split_thres)[0]

plt.plot(dist,'-o')
plt.show()

T_basemarker_base = robot.T_base_basemarker.inv()

split_id = np.append(0,split_id)
split_id = np.append(split_id,len(tool_T))

tool_T_align=[]
span=200
for sid in range(len(split_id)-1):
    id_range = int((split_id[sid]+split_id[sid+1])/2)
    id_range = np.arange(id_range-span,id_range+span).astype(int)
    
    this_rpy=[]
    this_p=[]
    for tid in id_range:
        this_T = Transform(q2R(tool_T[tid][3:7]),tool_T[tid][:3])
        this_T = T_basemarker_base*this_T
        this_rpy.append(R2rpy(this_T.R))
        this_p.append(this_T.p)
    this_rpy = np.mean(this_rpy,axis=0)
    this_p = np.mean(this_p,axis=0)
    thiq_p_q = np.append(this_p,R2q(rpy2R(this_rpy)))
    tool_T_align.append(thiq_p_q)
tool_T_align=np.array(tool_T_align)

remain_id=len(robot_q_align)%7
robot_q_align=robot_q_align[:-1*remain_id]
tool_T_align=tool_T_align[:-1*remain_id]
print('Total q:',len(robot_q_align),', total T:',len(tool_T_align))
np.savetxt(raw_data_dir+'_robot_q_align.csv',robot_q_align,delimiter=',')
np.savetxt(raw_data_dir+'_tool_T_align.csv',tool_T_align,delimiter=',')