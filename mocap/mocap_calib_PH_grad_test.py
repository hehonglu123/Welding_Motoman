import numpy as np
from copy import deepcopy
from general_robotics_toolbox import *
import pickle
import time
import sys
sys.path.append('../toolbox/')
from robot_def import *
from matplotlib import pyplot as plt

from qpsolvers import solve_qp

from numpy.random import default_rng
rng = default_rng()

Rx=np.array([1,0,0])
Ry=np.array([0,1,0])
Rz=np.array([0,0,1])

config_dir='../config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path='',d=0,\
pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
base_marker_config_file=config_dir+'MA2010_0516_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

T_base_basemarker = robot_weld.T_base_basemarker
T_basemarker_base = T_base_basemarker.inv()
robot_weld.T_tool_toolmarker=Transform(np.eye(3),[0,0,0])
robot_weld.robot.T_flange = robot_weld.T_tool_flange

PH_data_dir='PH_grad_data/test0516_R1/train_data_'
test_data_dir='kinematic_raw_data/test0516/'

test_robot_q = np.loadtxt(test_data_dir+'robot_q_align.csv',delimiter=',')
test_mocap_T = np.loadtxt(test_data_dir+'mocap_T_align.csv',delimiter=',')
assert len(test_robot_q)==len(test_mocap_T), f"Need to have the same amount of robot_q and mocap_T"

with open(PH_data_dir+'calib_PH_q.pickle','rb') as file:
    PH_q=pickle.load(file)
#### all train data q
train_q = []
training_error=[]
for qkey in PH_q.keys():
    train_q.append(np.array(qkey))
    training_error.append(PH_q[qkey]['train_pos_error'])
train_q=np.array(train_q)
training_error=np.array(training_error)
#####################

#### using zero config PH ####
train_q_zero_index = np.argmin(np.linalg.norm(train_q-np.zeros(2),ord=2,axis=1))
train_q_zero_key = tuple(train_q[train_q_zero_index])
qzero_P = PH_q[train_q_zero_key]['P']
qzero_H = PH_q[train_q_zero_key]['H']
#############################

#### using rotation PH (at zero) as baseline ####
baseline_P = deepcopy(robot_weld.calib_P)
baseline_H = deepcopy(robot_weld.calib_H)
#####################################

total_test_N = len(test_robot_q)

#### Gradient
plot_error=False
error_pos = []
error_ori = []
error_pos_baseline = []
error_ori_baseline = []
error_pos_PHZero = []
error_ori_PHZero = []
q2q3=[]
q1_all=[]
pos_all=[]
for N in range(total_test_N):
    test_q = test_robot_q[N]
    train_q_index = np.argmin(np.linalg.norm(train_q-test_q[1:3],ord=2,axis=1))
    train_q_key = tuple(train_q[train_q_index])

    # print("Test q2q3:",np.round(np.degrees(test_q[1:3]),3))
    # print("Using Train q2q3:",np.round(np.degrees(train_q[train_q_index]),3))

    T_marker_base = Transform(q2R(test_mocap_T[N][3:]),test_mocap_T[N][:3])
    T_tool_base = T_marker_base*robot_weld.T_tool_toolmarker

    #### get error
    opt_P = PH_q[train_q_key]['P']
    opt_H = PH_q[train_q_key]['H']
    robot_weld.robot.P=deepcopy(opt_P)
    robot_weld.robot.H=deepcopy(opt_H)
    robot_T = robot_weld.fwd(test_q)
    k,theta = R2rot(robot_T.R.T@T_tool_base.R)
    k=np.array(k)
    error_pos.append(T_tool_base.p-robot_T.p)
    error_ori.append(k*np.degrees(theta))

    #### get error (zero)
    robot_weld.robot.P=deepcopy(qzero_P)
    robot_weld.robot.H=deepcopy(qzero_H)
    robot_T = robot_weld.fwd(test_q)
    k,theta = R2rot(robot_T.R.T@T_tool_base.R)
    k=np.array(k)
    error_pos_PHZero.append(T_tool_base.p-robot_T.p)
    error_ori_PHZero.append(k*np.degrees(theta))

    #### get error (baseline)
    robot_weld.robot.P=deepcopy(baseline_P)
    robot_weld.robot.H=deepcopy(baseline_H)
    robot_T = robot_weld.fwd(test_q)
    k,theta = R2rot(robot_T.R.T@T_tool_base.R)
    k=np.array(k)
    error_pos_baseline.append(T_tool_base.p-robot_T.p)
    error_ori_baseline.append(k*np.degrees(theta))

    q2q3.append(np.degrees(test_q[1]+-1*test_q[2]))
    q1_all.append(np.degrees(test_q[0]))
    pos_all.append(str(round(T_tool_base.p[0]))+'\n'+str(round(T_tool_base.p[1]))+'\n'\
                   +str(round(T_tool_base.p[2])))
q2q3=np.array(q2q3)
q1_all=np.array(q1_all)
sort_q2q3_id = np.argsort(q2q3)
q2q3=q2q3[sort_q2q3_id]
error_pos_norm=np.linalg.norm(error_pos,ord=2,axis=1).flatten()
error_pos_PHZero_norm=np.linalg.norm(error_pos_PHZero,ord=2,axis=1).flatten()
error_pos_baseline_norm=np.linalg.norm(error_pos_baseline,ord=2,axis=1).flatten()

plt.plot(error_pos_norm,'-o',markersize=1,label='Opt PH')
plt.plot(error_pos_PHZero_norm,'-o',markersize=1,label='Zero PH')
plt.plot(error_pos_baseline_norm,'-o',markersize=1,label='Rotation PH')
plt.legend()
plt.title("Position Error using PH from Nearest q2q3")
# plt.xticks(np.arange(0,total_test_N,100),np.round(q1_all[::100]))
# plt.xlabel("J1 Angle at each Pose (degrees)")
plt.xticks(np.arange(0,total_test_N,50),pos_all[::50])
plt.xlabel("TCP Cartesian Position at poses")
plt.ylabel("Position Error (mm)")
plt.show()

print("Testing Data")
markdown_str=''
markdown_str+='||Mean (mm)|Std (mm)|Max (mm)|\n'
markdown_str+='|-|-|-|-|\n'
markdown_str+='|Rotate PH|'+format(round(np.mean(error_pos_baseline_norm),4),'.4f')+'|'+\
    format(round(np.std(error_pos_baseline_norm),4),'.4f')+'|'+format(round(np.max(error_pos_baseline_norm),4),'.4f')+'|\n'
markdown_str+='|Zero PH|'+format(round(np.mean(error_pos_PHZero_norm),4),'.4f')+'|'+\
    format(round(np.std(error_pos_PHZero_norm),4),'.4f')+'|'+format(round(np.max(error_pos_PHZero_norm),4),'.4f')+'|\n'
markdown_str+='|Optimize PH|'+format(round(np.mean(error_pos_norm),4),'.4f')+'|'+\
    format(round(np.std(error_pos_norm),4),'.4f')+'|'+format(round(np.max(error_pos_norm),4),'.4f')+'|\n'
print(markdown_str)

print("Training Data")
markdown_str=''
markdown_str+='||Mean (mm)|Std (mm)|Max (mm)|\n'
markdown_str+='|-|-|-|-|\n'
markdown_str+='|Rotate PH|'+format(round(np.mean(training_error),4),'.4f')+'|'+\
    format(round(np.std(training_error),4),'.4f')+'|'+format(round(np.max(training_error),4),'.4f')+'|\n'
print(markdown_str)