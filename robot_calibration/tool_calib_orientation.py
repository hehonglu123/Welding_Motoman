import numpy as np
from copy import deepcopy
from general_robotics_toolbox import *
from pathlib import Path
import glob
import pickle

import sys
sys.path.append('../toolbox/')
sys.path.append('../mocap/')
from robot_def import *
from PH_interp import *
from matplotlib import pyplot as plt
from dx200_motion_program_exec_client import *

def find_ptool(robot,all_q,ph_param=None):
    
    robot_Ts=[]
    for q in all_q:
        if ph_param:
            robot_T=robot.fwd_ph(q,ph_param)
        else:
            robot_T=robot.fwd(q)
        robot_Ts.append(H_from_RT(robot_T.R,robot_T.p))
    A=[]
    b=[]
    num_js=len(all_q)
    for i in range(num_js):
        next_id=i+1
        if next_id>=num_js:
            next_id=0

        A.extend(robot_Ts[i][:3,:3]-robot_Ts[next_id][:3,:3])
        b.extend(robot_Ts[next_id][:3,-1]-robot_Ts[i][:3,-1])

    p_tool=np.linalg.pinv(A)@b
    return p_tool

# data_dir='tool_data/'+'R1_weldgun_0809/'
data_dir='tool_data/'+'R2_mti_0810/'

ph_dataset_date='0804'

config_dir='../config/'

robot_type = 'R2'

PH_data_dir='../mocap/PH_grad_data/test'+ph_dataset_date+'_'+robot_type+'/'

if robot_type == 'R1':
    robot_marker_dir=config_dir+'MA2010_marker_config/'
    tool_marker_dir=config_dir+'weldgun_marker_config/'
    robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',\
                        tool_file_path=config_dir+'torch.csv',d=15,\
                        #  tool_file_path='',d=0,\
                        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
                        base_marker_config_file=robot_marker_dir+'MA2010_'+ph_dataset_date+'_marker_config.yaml',\
                        tool_marker_config_file=tool_marker_dir+'weldgun_'+ph_dataset_date+'_marker_config.yaml')
    nom_P=np.array([[0,0,0],[150,0,0],[0,0,760],\
                   [1082,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
    nom_H=np.array([[0,0,1],[0,1,0],[0,-1,0],\
                   [-1,0,0],[0,-1,0],[-1,0,0]]).T
elif robot_type == 'R2':
    robot_marker_dir=config_dir+'MA1440_marker_config/'
    tool_marker_dir=config_dir+'mti_marker_config/'
    robot=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',\
                        tool_file_path=config_dir+'mti.csv',\
                        pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
                        base_marker_config_file=robot_marker_dir+'MA1440_'+ph_dataset_date+'_marker_config.yaml',\
                        tool_marker_config_file=tool_marker_dir+'mti_'+ph_dataset_date+'_marker_config.yaml')
    nom_P=np.array([[0,0,0],[155,0,0],[0,0,614],\
                   [640,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
    nom_H=np.array([[0,0,1],[0,1,0],[0,-1,0],\
                   [-1,0,0],[0,-1,0],[-1,0,0]]).T

with open(PH_data_dir+'train_data_calib_PH_q.pickle','rb') as file:
    PH_q=pickle.load(file)
ph_param=PH_Param(nom_P,nom_H)
ph_param.fit(PH_q,method='FBF')

origin_R_tool=deepcopy(robot.robot.R_tool)
robot.robot.R_tool = np.eye(3)
robot.robot.p_tool = np.zeros(3)

# find origin
num_js=len(glob.glob(data_dir+'pose_js_origin_*.csv'))
all_q=[]
for i in range(num_js):
    q=np.loadtxt(data_dir+'pose_js_origin_'+str(i)+'.csv',delimiter=',')
    all_q.append(q)
p_origin=find_ptool(robot,all_q,ph_param)

# find z-axis
num_js=len(glob.glob(data_dir+'pose_js_zaxis_*.csv'))
all_q=[]
for i in range(num_js):
    q=np.loadtxt(data_dir+'pose_js_zaxis_'+str(i)+'.csv',delimiter=',')
    all_q.append(q)
p_zaxis=find_ptool(robot,all_q,ph_param)

# find x-axis
num_js=len(glob.glob(data_dir+'pose_js_xaxis_*.csv'))
all_q=[]
for i in range(num_js):
    q=np.loadtxt(data_dir+'pose_js_xaxis_'+str(i)+'.csv',delimiter=',')
    all_q.append(q)
p_xaxis=find_ptool(robot,all_q,ph_param)

z_axix=-1*(p_zaxis-p_origin)
z_axix=z_axix/np.linalg.norm(z_axix)
x_axis=-1*(p_xaxis-p_origin)
x_axis=x_axis/np.linalg.norm(x_axis)
x_axis=x_axis-np.dot(x_axis,z_axix)*z_axix
x_axis=x_axis/np.linalg.norm(x_axis)
y_axis=np.cross(z_axix,x_axis)
y_axis=y_axis/np.linalg.norm(y_axis)

R_tool = np.array([x_axis,y_axis,z_axix])

print(p_origin)
print(R_tool)

if robot_type=='R1':
    p_origin = p_origin+-1*R_tool[:,-1]*15
elif robot_type=='R2':
    p_origin = p_origin+-1*R_tool[:,-1]*120
    
print(p_origin)

toolT=H_from_RT(R_tool,p_origin)

np.savetxt(data_dir+'tool.csv',toolT,delimiter=',')