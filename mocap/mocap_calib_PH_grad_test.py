import numpy as np
from copy import deepcopy
from general_robotics_toolbox import *
import pickle
import time
import sys
sys.path.append('../toolbox/')
from robot_def import *
from matplotlib import pyplot as plt
from PH_interp import *

from numpy.random import default_rng
rng = default_rng()

Rx=np.array([1,0,0])
Ry=np.array([0,1,0])
Rz=np.array([0,0,1])

config_dir='../config/'
# robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path='',d=0,\
# pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
# base_marker_config_file=config_dir+'MA2010_0516_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')
### z pointing x-axis (with 22 deg angle), y pointing y-axis
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',d=15,\
pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

# print(robot_weld.fwd(np.zeros(6)))
# exit()

T_base_basemarker = robot_weld.T_base_basemarker
T_basemarker_base = T_base_basemarker.inv()
robot_weld.robot.T_flange = robot_weld.T_tool_flange

#### using rigid body
# robot_weld.T_tool_toolmarker=Transform(np.eye(3),[0,0,0])
#### using tool
robot_weld.robot.R_tool = robot_weld.T_tool_toolmarker.R
robot_weld.robot.p_tool = robot_weld.T_tool_toolmarker.p

PH_data_dir='PH_grad_data/test0516_R1/train_data_'
test_data_dir='kinematic_raw_data/test0516/'

test_robot_q = np.loadtxt(test_data_dir+'robot_q_align.csv',delimiter=',')
test_mocap_T = np.loadtxt(test_data_dir+'mocap_T_align.csv',delimiter=',')
assert len(test_robot_q)==len(test_mocap_T), f"Need to have the same amount of robot_q and mocap_T"

with open(PH_data_dir+'calib_PH_q_torch.pickle','rb') as file:
    PH_q=pickle.load(file)
with open(PH_data_dir+'calib_one_PH_torch.pickle','rb') as file:
    PH_q_one=pickle.load(file)
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

#### one PH for all ####
universal_P = PH_q_one['P']
universal_H = PH_q_one['H']
training_error_universal=PH_q_one['train_pos_error']
# plt.plot(np.mean(training_error_universal,axis=1))
# plt.xlabel("Iteration")
# plt.ylabel("Average Position Error Norm (mm)")
# plt.title("Average Position error norm of all poses")
# plt.show()
########################

#### using rotation PH (at zero) as baseline ####
baseline_P = deepcopy(robot_weld.calib_P)
baseline_H = deepcopy(robot_weld.calib_H)
#####################################

total_test_N = len(test_robot_q)

## PH_param
ph_param_near=PH_Param()
ph_param_near.fit(PH_q,method='nearest')
ph_param_lin=PH_Param()
ph_param_lin.fit(PH_q,method='linear')
ph_param_cub=PH_Param()
ph_param_cub.fit(PH_q,method='cubic')
ph_param_rbf=PH_Param()
ph_param_rbf.fit(PH_q,method='RBF')
# exit()

#### Gradient
plot_error=False
error_pos_near = []
error_ori_near = []
error_pos_lin = []
error_ori_lin = []
error_pos_cub = []
error_ori_cub = []
error_pos_rbf = []
error_ori_rbf = []
error_pos_baseline = []
error_ori_baseline = []
error_pos_PHZero = []
error_ori_PHZero = []
error_pos_onePH = []
error_ori_onePH = []
q2q3=[]
q1_all=[]
pos_all=[]
for N in range(total_test_N):
    test_q = test_robot_q[N]
    # print("Test q2q3:",np.round(np.degrees(test_q[1:3]),3))
    # print("Using Train q2q3:",np.round(np.degrees(train_q[train_q_index]),3))

    T_marker_base = Transform(q2R(test_mocap_T[N][3:]),test_mocap_T[N][:3])
    T_tool_base = T_marker_base*robot_weld.T_tool_toolmarker

    #### get error (nearest)
    opt_P,opt_H = ph_param_near.predict(test_q[1:3])
    robot_weld.robot.P=deepcopy(opt_P)
    robot_weld.robot.H=deepcopy(opt_H)
    robot_T = robot_weld.fwd(test_q)
    k,theta = R2rot(robot_T.R.T@T_tool_base.R)
    k=np.array(k)
    error_pos_near.append(T_tool_base.p-robot_T.p)
    error_ori_near.append(k*np.degrees(theta))

    #### get error (linear)
    opt_P,opt_H = ph_param_lin.predict(test_q[1:3])
    if np.any(opt_P is np.nan) or np.any(opt_H is np.nan):
        print(np.degrees(test_q))
    robot_weld.robot.P=deepcopy(opt_P)
    robot_weld.robot.H=deepcopy(opt_H)
    robot_T = robot_weld.fwd(test_q)
    k,theta = R2rot(robot_T.R.T@T_tool_base.R)
    k=np.array(k)
    error_pos_lin.append(T_tool_base.p-robot_T.p)
    error_ori_lin.append(k*np.degrees(theta))

    #### get error (cubic)
    opt_P,opt_H = ph_param_cub.predict(test_q[1:3])
    if np.any(opt_P is np.nan) or np.any(opt_H is np.nan):
        print(np.degrees(test_q))
    robot_weld.robot.P=deepcopy(opt_P)
    robot_weld.robot.H=deepcopy(opt_H)
    robot_T = robot_weld.fwd(test_q)
    k,theta = R2rot(robot_T.R.T@T_tool_base.R)
    k=np.array(k)
    error_pos_cub.append(T_tool_base.p-robot_T.p)
    error_ori_cub.append(k*np.degrees(theta))

    #### get error (rbf)
    opt_P,opt_H = ph_param_rbf.predict(test_q[1:3])
    if np.any(opt_P is np.nan) or np.any(opt_H is np.nan):
        print(np.degrees(test_q))
    robot_weld.robot.P=deepcopy(opt_P)
    robot_weld.robot.H=deepcopy(opt_H)
    robot_T = robot_weld.fwd(test_q)
    k,theta = R2rot(robot_T.R.T@T_tool_base.R)
    k=np.array(k)
    error_pos_rbf.append(T_tool_base.p-robot_T.p)
    error_ori_rbf.append(k*np.degrees(theta))

    #### get error (zero)
    robot_weld.robot.P=deepcopy(qzero_P)
    robot_weld.robot.H=deepcopy(qzero_H)
    robot_T = robot_weld.fwd(test_q)
    k,theta = R2rot(robot_T.R.T@T_tool_base.R)
    k=np.array(k)
    error_pos_PHZero.append(T_tool_base.p-robot_T.p)
    error_ori_PHZero.append(k*np.degrees(theta))

    #### get error (one PH)
    robot_weld.robot.P=deepcopy(universal_P)
    robot_weld.robot.H=deepcopy(universal_H)
    robot_T = robot_weld.fwd(test_q)
    k,theta = R2rot(robot_T.R.T@T_tool_base.R)
    k=np.array(k)
    error_pos_onePH.append(T_tool_base.p-robot_T.p)
    error_ori_onePH.append(k*np.degrees(theta))

    #### get error (baseline)
    robot_weld.robot.P=deepcopy(baseline_P)
    robot_weld.robot.H=deepcopy(baseline_H)
    robot_T = robot_weld.fwd(test_q)
    k,theta = R2rot(robot_T.R.T@T_tool_base.R)
    k=np.array(k)
    error_pos_baseline.append(T_tool_base.p-robot_T.p)
    error_ori_baseline.append(k*np.degrees(theta))

    # if np.all(test_q-np.zeros(6)<1e-3):
    #     print(robot_T)
    #     print(T_tool_base)
    #     exit()

    q2q3.append(np.degrees(test_q[1]+-1*test_q[2]))
    q1_all.append(np.degrees(test_q[0]))
    pos_all.append(str(round(T_tool_base.p[0]))+'\n'+str(round(T_tool_base.p[1]))+'\n'\
                   +str(round(T_tool_base.p[2])))
q2q3=np.array(q2q3)
q1_all=np.array(q1_all)
sort_q2q3_id = np.argsort(q2q3)
q2q3=q2q3[sort_q2q3_id]
error_pos_near_norm=np.linalg.norm(error_pos_near,ord=2,axis=1).flatten()
error_pos_lin_norm=np.linalg.norm(error_pos_lin,ord=2,axis=1).flatten()
error_pos_cub_norm=np.linalg.norm(error_pos_cub,ord=2,axis=1).flatten()
error_pos_rbf_norm=np.linalg.norm(error_pos_rbf,ord=2,axis=1).flatten()
error_pos_PHZero_norm=np.linalg.norm(error_pos_PHZero,ord=2,axis=1).flatten()
error_pos_onePH_norm=np.linalg.norm(error_pos_onePH,ord=2,axis=1).flatten()
error_pos_baseline_norm=np.linalg.norm(error_pos_baseline,ord=2,axis=1).flatten()

error_ori_near_norm=np.linalg.norm(error_ori_near,ord=2,axis=1).flatten()
error_ori_lin_norm=np.linalg.norm(error_ori_lin,ord=2,axis=1).flatten()
error_ori_cub_norm=np.linalg.norm(error_ori_cub,ord=2,axis=1).flatten()
error_ori_rbf_norm=np.linalg.norm(error_ori_rbf,ord=2,axis=1).flatten()
error_ori_PHZero_norm=np.linalg.norm(error_ori_PHZero,ord=2,axis=1).flatten()
error_ori_onePH_norm=np.linalg.norm(error_ori_onePH,ord=2,axis=1).flatten()
error_ori_baseline_norm=np.linalg.norm(error_ori_baseline,ord=2,axis=1).flatten()

plt.plot(error_pos_baseline_norm,'-o',markersize=1,label='Rotation PH')
plt.plot(error_pos_PHZero_norm,'-o',markersize=1,label='Zero PH')
plt.plot(error_pos_onePH_norm,'-o',markersize=1,label='One PH')
plt.plot(error_pos_near_norm,'-o',markersize=1,label='Nearest PH')
plt.plot(error_pos_lin_norm,'-o',markersize=1,label='Linear Interp PH')
plt.plot(error_pos_cub_norm,'-o',markersize=1,label='Cubic Interp PH')
plt.plot(error_pos_rbf_norm,'-o',markersize=1,label='RBF Interp PH')
plt.legend()
plt.title("Position Error using Optimized PH")
# plt.xticks(np.arange(0,total_test_N,100),np.round(q1_all[::100]))
# plt.xlabel("J1 Angle at each Pose (degrees)")
plt.xticks(np.arange(0,total_test_N,50),pos_all[::50])
plt.xlabel("TCP Cartesian Position at Poses")
plt.ylabel("Position Error (mm)")
plt.show()

plt.plot(error_ori_baseline_norm,'-o',markersize=1,label='Rotation PH')
plt.plot(error_ori_PHZero_norm,'-o',markersize=1,label='Zero PH')
plt.plot(error_ori_onePH_norm,'-o',markersize=1,label='One PH')
plt.plot(error_ori_near_norm,'-o',markersize=1,label='Nearest PH')
plt.plot(error_ori_lin_norm,'-o',markersize=1,label='Linear Interp PH')
plt.plot(error_ori_cub_norm,'-o',markersize=1,label='Cubic Interp PH')
plt.plot(error_ori_rbf_norm,'-o',markersize=1,label='RBF Interp PH')
plt.legend()
plt.title("Orientation Error using Optimized PH")
# plt.xticks(np.arange(0,total_test_N,100),np.round(q1_all[::100]))
# plt.xlabel("J1 Angle at each orie (degrees)")
plt.xticks(np.arange(0,total_test_N,50),pos_all[::50])
plt.xlabel("TCP Cartesian Position at Poses")
plt.ylabel("Orientation Error (deg)")
plt.show()

print("Testing Data (Position)")
markdown_str=''
markdown_str+='||Mean (mm)|Std (mm)|Max (mm)|\n'
markdown_str+='|-|-|-|-|\n'
markdown_str+='|Rotate PH|'+format(round(np.mean(error_pos_baseline_norm),4),'.4f')+'|'+\
    format(round(np.std(error_pos_baseline_norm),4),'.4f')+'|'+format(round(np.max(error_pos_baseline_norm),4),'.4f')+'|\n'
markdown_str+='|Zero PH|'+format(round(np.mean(error_pos_PHZero_norm),4),'.4f')+'|'+\
    format(round(np.std(error_pos_PHZero_norm),4),'.4f')+'|'+format(round(np.max(error_pos_PHZero_norm),4),'.4f')+'|\n'
markdown_str+='|One PH|'+format(round(np.mean(error_pos_onePH_norm),4),'.4f')+'|'+\
    format(round(np.std(error_pos_onePH_norm),4),'.4f')+'|'+format(round(np.max(error_pos_onePH_norm),4),'.4f')+'|\n'
markdown_str+='|Nearest PH|'+format(round(np.mean(error_pos_near_norm),4),'.4f')+'|'+\
    format(round(np.std(error_pos_near_norm),4),'.4f')+'|'+format(round(np.max(error_pos_near_norm),4),'.4f')+'|\n'
markdown_str+='|Linear Interp PH|'+format(round(np.mean(error_pos_lin_norm),4),'.4f')+'|'+\
    format(round(np.std(error_pos_lin_norm),4),'.4f')+'|'+format(round(np.max(error_pos_lin_norm),4),'.4f')+'|\n'
markdown_str+='|Cubic Interp PH|'+format(round(np.mean(error_pos_cub_norm),4),'.4f')+'|'+\
    format(round(np.std(error_pos_cub_norm),4),'.4f')+'|'+format(round(np.max(error_pos_cub_norm),4),'.4f')+'|\n'
markdown_str+='|RBF Interp PH|'+format(round(np.mean(error_pos_rbf_norm),4),'.4f')+'|'+\
    format(round(np.std(error_pos_rbf_norm),4),'.4f')+'|'+format(round(np.max(error_pos_rbf_norm),4),'.4f')+'|\n'
print(markdown_str)

print("Testing Data (Orientation)")
markdown_str=''
markdown_str+='||Mean (deg)|Std (deg)|Max (deg)|\n'
markdown_str+='|-|-|-|-|\n'
markdown_str+='|Rotate PH|'+format(round(np.mean(error_ori_baseline_norm),4),'.4f')+'|'+\
    format(round(np.std(error_ori_baseline_norm),4),'.4f')+'|'+format(round(np.max(error_ori_baseline_norm),4),'.4f')+'|\n'
markdown_str+='|Zero PH|'+format(round(np.mean(error_ori_PHZero_norm),4),'.4f')+'|'+\
    format(round(np.std(error_ori_PHZero_norm),4),'.4f')+'|'+format(round(np.max(error_ori_PHZero_norm),4),'.4f')+'|\n'
markdown_str+='|One PH|'+format(round(np.mean(error_ori_onePH_norm),4),'.4f')+'|'+\
    format(round(np.std(error_ori_onePH_norm),4),'.4f')+'|'+format(round(np.max(error_ori_onePH_norm),4),'.4f')+'|\n'
markdown_str+='|Nearest PH|'+format(round(np.mean(error_ori_near_norm),4),'.4f')+'|'+\
    format(round(np.std(error_ori_near_norm),4),'.4f')+'|'+format(round(np.max(error_ori_near_norm),4),'.4f')+'|\n'
markdown_str+='|Linear Interp PH|'+format(round(np.mean(error_ori_lin_norm),4),'.4f')+'|'+\
    format(round(np.std(error_ori_lin_norm),4),'.4f')+'|'+format(round(np.max(error_ori_lin_norm),4),'.4f')+'|\n'
markdown_str+='|Cubic Interp PH|'+format(round(np.mean(error_ori_cub_norm),4),'.4f')+'|'+\
    format(round(np.std(error_ori_cub_norm),4),'.4f')+'|'+format(round(np.max(error_ori_cub_norm),4),'.4f')+'|\n'
markdown_str+='|RBF Interp PH|'+format(round(np.mean(error_ori_rbf_norm),4),'.4f')+'|'+\
    format(round(np.std(error_ori_rbf_norm),4),'.4f')+'|'+format(round(np.max(error_ori_rbf_norm),4),'.4f')+'|\n'
print(markdown_str)

print("Training Data")
markdown_str=''
markdown_str+='||Mean (mm)|Std (mm)|Max (mm)|\n'
markdown_str+='|-|-|-|-|\n'
markdown_str+='|One PH|'+format(round(np.mean(training_error_universal[-1]),4),'.4f')+'|'+\
    format(round(np.std(training_error_universal[-1]),4),'.4f')+'|'+format(round(np.max(training_error_universal[-1]),4),'.4f')+'|\n'
markdown_str+='|Optimize PH|'+format(round(np.mean(training_error),4),'.4f')+'|'+\
    format(round(np.std(training_error),4),'.4f')+'|'+format(round(np.max(training_error),4),'.4f')+'|\n'
print(markdown_str)