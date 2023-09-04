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

ph_dataset_date='0801'
test_dataset_date='0801'
config_dir='../config/'

robot_type = 'R1'

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

T_base_basemarker = robot.T_base_basemarker
T_basemarker_base = T_base_basemarker.inv()

#### using rigid body
use_toolmaker=True
T_base_basemarker = robot.T_base_basemarker
T_basemarker_base = T_base_basemarker.inv()

if use_toolmaker:
    robot.robot.R_tool = robot.T_toolmarker_flange.R
    robot.robot.p_tool = robot.T_toolmarker_flange.p
    robot.T_tool_toolmarker = Transform(np.eye(3),[0,0,0])
    
    # robot.robot.R_tool = np.eye(3)
    # robot.robot.p_tool = np.zeros(3)
    # robot.T_tool_toolmarker = robot.T_toolmarker_flange.inv()

PH_data_dir='PH_grad_data/test'+ph_dataset_date+'_'+robot_type+'/train_data_'
# test_data_dir='kinematic_raw_data/test'+test_dataset_date+'_aftercalib/'
test_data_dir='kinematic_raw_data/test'+test_dataset_date+'_'+robot_type+'/'

print(PH_data_dir)
print(test_data_dir)

use_raw=False
test_robot_q = np.loadtxt(test_data_dir+'robot_q_align.csv',delimiter=',')
test_mocap_T = np.loadtxt(test_data_dir+'mocap_T_align.csv',delimiter=',')
if use_raw:
    test_mocap_T=[]
    toolrigid_raw = np.loadtxt(test_data_dir+'mocap_tool_T_raw.csv',delimiter=',')
    baserigid_raw = np.loadtxt(test_data_dir+'mocap_base_T_raw.csv',delimiter=',')
    
    raw_id=0
    same_pose_thres=0.1 #mm
    pos_toolrigid=[]
    pose_toolrigid_base=[]
    while True:
        if len(pos_toolrigid)==0:
            pos_toolrigid.append(toolrigid_raw[raw_id][:3])
        elif np.linalg.norm(np.mean(pos_toolrigid,axis=0)-toolrigid_raw[raw_id][:3])<same_pose_thres:
            pos_toolrigid.append(toolrigid_raw[raw_id][:3])
        else:
            pose_toolrigid_base=np.array(pose_toolrigid_base)
            if np.linalg.norm(np.degrees(np.max(pose_toolrigid_base[:,3:],axis=0)\
                -np.min(pose_toolrigid_base[:,3:],axis=0)))>0.1:
                print("RPY max min:",np.degrees(np.min(pose_toolrigid_base[:,3:],axis=0)),\
                    np.degrees(np.max(pose_toolrigid_base[:,3:],axis=0)))
            p_mean = np.mean(pose_toolrigid_base,axis=0)
            this_T = np.append(p_mean[:3],R2q(rpy2R(p_mean[3:])))
            test_mocap_T.append(this_T)
            pos_toolrigid=[]
            pose_toolrigid_base=[]
            continue
        
        T_mocap_basemarker = Transform(q2R(baserigid_raw[raw_id][3:]),baserigid_raw[raw_id][:3]).inv()
        T_marker_mocap = Transform(q2R(toolrigid_raw[raw_id][3:]),toolrigid_raw[raw_id][:3])
        T_marker_basemarker = T_mocap_basemarker*T_marker_mocap
        T_marker_base = T_basemarker_base*T_marker_basemarker
        pose_toolrigid_base.append(np.append(T_marker_base.p,R2rpy(T_marker_base.R)))
        raw_id+=1
        
        if raw_id>=len(toolrigid_raw):
            pose_toolrigid_base=np.array(pose_toolrigid_base)
            if np.linalg.norm(np.degrees(np.max(pose_toolrigid_base[:,3:],axis=0)\
                -np.min(pose_toolrigid_base[:,3:],axis=0)))>0.1:
                print("RPY max min:",np.degrees(np.min(pose_toolrigid_base[:,3:],axis=0)),\
                    np.degrees(np.max(pose_toolrigid_base[:,3:],axis=0)))
            p_mean = np.mean(pose_toolrigid_base,axis=0)
            this_T = np.append(p_mean[:3],R2q(rpy2R(p_mean[3:])))
            test_mocap_T.append(this_T)
            pos_toolrigid=[]
            break


print(len(test_mocap_T))
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
try:
    training_error=np.array(training_error)
except:
    training_error_last=[]
    for te in training_error:
        training_error_last=te[-1]
    training_error=training_error_last
#####################

#### using zero config PH ####
train_q_zero_index = np.argmin(np.linalg.norm(train_q-np.zeros(2),ord=2,axis=1))
train_q_zero_key = tuple(train_q[train_q_zero_index])
qzero_P = PH_q[train_q_zero_key]['P']
qzero_H = PH_q[train_q_zero_key]['H']
#############################

try:
    with open(PH_data_dir+'calib_one_PH.pickle','rb') as file:
        PH_q_one=pickle.load(file)
except:
    PH_q_one=PH_q[train_q_zero_key]
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

#### pre-calib #####################
origin_P = deepcopy(robot.robot.P)
origin_H = deepcopy(robot.robot.H)

#### using rotation PH (at zero) as baseline ####
baseline_P = deepcopy(robot.calib_P)
baseline_H = deepcopy(robot.calib_H)
#####################################

total_test_N = len(test_robot_q)

## PH_param
ph_param_near=PH_Param(nom_P,nom_H)
ph_param_near.fit(PH_q,method='nearest')
ph_param_lin=PH_Param(nom_P,nom_H)
ph_param_lin.fit(PH_q,method='linear')
ph_param_cub=PH_Param(nom_P,nom_H)
ph_param_cub.fit(PH_q,method='cubic')
ph_param_rbf=PH_Param(nom_P,nom_H)
ph_param_rbf.fit(PH_q,method='RBF')
ph_param_fbf=PH_Param(nom_P,nom_H)
ph_param_fbf.fit(PH_q,method='FBF')
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
error_pos_fbf = []
error_ori_fbf = []
error_pos_baseline = []
error_ori_baseline = []
error_pos_PHZero = []
error_ori_PHZero = []
error_pos_onePH = []
error_ori_onePH = []
error_pos_origin = []
error_ori_origin = []
q2q3=[]
q1_all=[]
pos_all=[]
for N in range(total_test_N):
    test_q = test_robot_q[N]
    # print("Test q2q3:",np.round(np.degrees(test_q[1:3]),3))
    # print("Using Train q2q3:",np.round(np.degrees(train_q[train_q_index]),3))

    T_marker_base = Transform(q2R(test_mocap_T[N][3:]),test_mocap_T[N][:3])
    T_tool_base = T_marker_base*robot.T_tool_toolmarker

    #### get error (nearest)
    opt_P,opt_H = ph_param_near.predict(test_q[1:3])
    robot.robot.P=deepcopy(opt_P)
    robot.robot.H=deepcopy(opt_H)
    robot_T = robot.fwd(test_q)
    k,theta = R2rot(robot_T.R.T@T_tool_base.R)
    k=np.array(k)
    error_pos_near.append(T_tool_base.p-robot_T.p)
    error_ori_near.append(k*np.degrees(theta))

    #### get error (linear)
    opt_P,opt_H = ph_param_lin.predict(test_q[1:3])
    if np.any(opt_P is np.nan) or np.any(opt_H is np.nan):
        print(np.degrees(test_q))
    robot.robot.P=deepcopy(opt_P)
    robot.robot.H=deepcopy(opt_H)
    robot_T = robot.fwd(test_q)
    k,theta = R2rot(robot_T.R.T@T_tool_base.R)
    k=np.array(k)
    error_pos_lin.append(T_tool_base.p-robot_T.p)
    error_ori_lin.append(k*np.degrees(theta))

    #### get error (cubic)
    opt_P,opt_H = ph_param_cub.predict(test_q[1:3])
    if np.any(opt_P is np.nan) or np.any(opt_H is np.nan):
        print(np.degrees(test_q))
    robot.robot.P=deepcopy(opt_P)
    robot.robot.H=deepcopy(opt_H)
    robot_T = robot.fwd(test_q)
    k,theta = R2rot(robot_T.R.T@T_tool_base.R)
    k=np.array(k)
    error_pos_cub.append(T_tool_base.p-robot_T.p)
    error_ori_cub.append(k*np.degrees(theta))

    #### get error (rbf)
    opt_P,opt_H = ph_param_rbf.predict(test_q[1:3])
    if np.any(opt_P is np.nan) or np.any(opt_H is np.nan):
        print(np.degrees(test_q))
    robot.robot.P=deepcopy(opt_P)
    robot.robot.H=deepcopy(opt_H)
    robot_T = robot.fwd(test_q)
    k,theta = R2rot(robot_T.R.T@T_tool_base.R)
    k=np.array(k)
    error_pos_rbf.append(T_tool_base.p-robot_T.p)
    error_ori_rbf.append(k*np.degrees(theta))
    
    #### get error (fbf)
    opt_P,opt_H = ph_param_fbf.predict(test_q[1:3])
    if np.any(opt_P is np.nan) or np.any(opt_H is np.nan):
        print(np.degrees(test_q))
    robot.robot.P=deepcopy(opt_P)
    robot.robot.H=deepcopy(opt_H)
    robot_T = robot.fwd(test_q)
    k,theta = R2rot(robot_T.R.T@T_tool_base.R)
    k=np.array(k)
    error_pos_fbf.append(T_tool_base.p-robot_T.p)
    error_ori_fbf.append(k*np.degrees(theta))

    #### get error (zero)
    robot.robot.P=deepcopy(qzero_P)
    robot.robot.H=deepcopy(qzero_H)
    robot_T = robot.fwd(test_q)
    k,theta = R2rot(robot_T.R.T@T_tool_base.R)
    k=np.array(k)
    error_pos_PHZero.append(T_tool_base.p-robot_T.p)
    error_ori_PHZero.append(k*np.degrees(theta))

    #### get error (one PH)
    robot.robot.P=deepcopy(universal_P)
    robot.robot.H=deepcopy(universal_H)
    robot_T = robot.fwd(test_q)
    k,theta = R2rot(robot_T.R.T@T_tool_base.R)
    k=np.array(k)
    error_pos_onePH.append(T_tool_base.p-robot_T.p)
    error_ori_onePH.append(k*np.degrees(theta))

    #### get error (baseline)
    robot.robot.P=deepcopy(baseline_P)
    robot.robot.H=deepcopy(baseline_H)
    robot_T = robot.fwd(test_q)
    k,theta = R2rot(robot_T.R.T@T_tool_base.R)
    k=np.array(k)
    error_pos_baseline.append(T_tool_base.p-robot_T.p)
    error_ori_baseline.append(k*np.degrees(theta))
    
    #### get error (origin)
    robot.robot.P=deepcopy(origin_P)
    robot.robot.H=deepcopy(origin_H)
    robot_T = robot.fwd(test_q)
    k,theta = R2rot(robot_T.R.T@T_tool_base.R)
    k=np.array(k)
    error_pos_origin.append(T_tool_base.p-robot_T.p)
    error_ori_origin.append(k*np.degrees(theta))

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
error_pos_fbf_norm=np.linalg.norm(error_pos_fbf,ord=2,axis=1).flatten()
error_pos_PHZero_norm=np.linalg.norm(error_pos_PHZero,ord=2,axis=1).flatten()
error_pos_onePH_norm=np.linalg.norm(error_pos_onePH,ord=2,axis=1).flatten()
error_pos_baseline_norm=np.linalg.norm(error_pos_baseline,ord=2,axis=1).flatten()
error_pos_origin_norm=np.linalg.norm(error_pos_origin,ord=2,axis=1).flatten()

error_ori_near_norm=np.linalg.norm(error_ori_near,ord=2,axis=1).flatten()
error_ori_lin_norm=np.linalg.norm(error_ori_lin,ord=2,axis=1).flatten()
error_ori_cub_norm=np.linalg.norm(error_ori_cub,ord=2,axis=1).flatten()
error_ori_rbf_norm=np.linalg.norm(error_ori_rbf,ord=2,axis=1).flatten()
error_ori_fbf_norm=np.linalg.norm(error_ori_fbf,ord=2,axis=1).flatten()
error_ori_PHZero_norm=np.linalg.norm(error_ori_PHZero,ord=2,axis=1).flatten()
error_ori_onePH_norm=np.linalg.norm(error_ori_onePH,ord=2,axis=1).flatten()
error_ori_baseline_norm=np.linalg.norm(error_ori_baseline,ord=2,axis=1).flatten()
error_ori_origin_norm=np.linalg.norm(error_ori_origin,ord=2,axis=1).flatten()

plot_origin=True
if plot_origin:
    plt.plot(error_pos_origin_norm,'-o',markersize=1,label='Origin PH')    
plt.plot(error_pos_baseline_norm,'-o',markersize=1,label='CPA PH')
# plt.plot(error_pos_PHZero_norm,'-o',markersize=1,label='Zero PH')
# plt.plot(error_pos_onePH_norm,'-o',markersize=1,label='One PH')
plt.plot(error_pos_near_norm,'-o',markersize=1,label='Nearest PH')
plt.plot(error_pos_lin_norm,'-o',markersize=1,label='Linear Interp PH')
# plt.plot(error_pos_cub_norm,'-o',markersize=1,label='Cubic Interp PH')
# plt.plot(error_pos_rbf_norm,'-o',markersize=1,label='RBF Interp PH')
plt.plot(error_pos_fbf_norm,'-o',markersize=1,label='Fourier Basis PH')
plt.legend(fontsize=22)
plt.title("Position Error using Optimized PH",fontsize=32)
# plt.xticks(np.arange(0,total_test_N,100),np.round(q1_all[::100]))
# plt.xlabel("J1 Angle at each Pose (degrees)")
# plt.xticks(np.arange(0,total_test_N,50),pos_all[::50])
plt.xticks(np.arange(0,total_test_N,100),fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel("Testing pose index",fontsize=26)
plt.ylabel("Position Error (mm)",fontsize=26)
plt.tight_layout()
plt.show()

if plot_origin:
    plt.plot(error_ori_origin_norm,'-o',markersize=1,label='Origin PH')
plt.plot(error_ori_baseline_norm,'-o',markersize=1,label='CPA PH')
# plt.plot(error_ori_PHZero_norm,'-o',markersize=1,label='Zero PH')
# plt.plot(error_ori_onePH_norm,'-o',markersize=1,label='One PH')
# plt.plot(error_ori_near_norm,'-o',markersize=1,label='Nearest PH')
plt.plot(error_ori_lin_norm,'-o',markersize=1,label='Linear Interp PH')
# plt.plot(error_ori_cub_norm,'-o',markersize=1,label='Cubic Interp PH')
# plt.plot(error_ori_rbf_norm,'-o',markersize=1,label='RBF Interp PH')
plt.plot(error_ori_fbf_norm,'-o',markersize=1,label='Fourier Basis PH')
plt.legend()
plt.title("Orientation Error using Optimized PH")
# plt.xticks(np.arange(0,total_test_N,100),np.round(q1_all[::100]))
# plt.xlabel("J1 Angle at each orie (degrees)")
plt.xticks(np.arange(0,total_test_N,50),pos_all[::50])
plt.xlabel("TCP Cartesian Position at Poses")
plt.ylabel("Orientation Error (deg)")
plt.tight_layout()
plt.show()

print("Testing Data (Position)")
markdown_str=''
markdown_str+='||Mean (mm)|Std (mm)|Max (mm)|\n'
markdown_str+='|-|-|-|-|\n'
markdown_str+='|Origin PH|'+format(round(np.mean(error_pos_origin_norm),4),'.4f')+'|'+\
    format(round(np.std(error_pos_origin_norm),4),'.4f')+'|'+format(round(np.max(error_pos_origin_norm),4),'.4f')+'|\n'
markdown_str+='|CPA PH|'+format(round(np.mean(error_pos_baseline_norm),4),'.4f')+'|'+\
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
markdown_str+='|FBF Interp PH|'+format(round(np.mean(error_pos_fbf_norm),4),'.4f')+'|'+\
    format(round(np.std(error_pos_fbf_norm),4),'.4f')+'|'+format(round(np.max(error_pos_fbf_norm),4),'.4f')+'|\n'
print(markdown_str)

print("Testing Data (Orientation)")
markdown_str=''
markdown_str+='||Mean (deg)|Std (deg)|Max (deg)|\n'
markdown_str+='|-|-|-|-|\n'
markdown_str+='|Origin PH|'+format(round(np.mean(error_ori_origin_norm),4),'.4f')+'|'+\
    format(round(np.std(error_ori_origin_norm),4),'.4f')+'|'+format(round(np.max(error_ori_origin_norm),4),'.4f')+'|\n'
markdown_str+='|CPA PH|'+format(round(np.mean(error_ori_baseline_norm),4),'.4f')+'|'+\
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
markdown_str+='|FBF Interp PH|'+format(round(np.mean(error_ori_fbf_norm),4),'.4f')+'|'+\
    format(round(np.std(error_ori_fbf_norm),4),'.4f')+'|'+format(round(np.max(error_ori_fbf_norm),4),'.4f')+'|\n'
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