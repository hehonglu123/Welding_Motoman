import numpy as np
from copy import deepcopy
from general_robotics_toolbox import *
import pickle
import time
import sys
sys.path.append('../toolbox/')
from robot_def import *
from matplotlib import pyplot as plt
from scipy.optimize import fminbound
from qpsolvers import solve_qp
from calib_analytic_grad import *

from numpy.random import default_rng
rng = default_rng()

Rx=np.array([1,0,0])
Ry=np.array([0,1,0])
Rz=np.array([0,0,1])

dataset_date='0804'

config_dir='../config/'

robot_type = 'R2'

if robot_type == 'R1':
    robot_marker_dir=config_dir+'MA2010_marker_config/'
    tool_marker_dir=config_dir+'weldgun_marker_config/'
    robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',\
                        tool_file_path=config_dir+'torch.csv',d=15,\
                        #  tool_file_path='',d=0,\
                        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
                        base_marker_config_file=robot_marker_dir+'MA2010_'+dataset_date+'_marker_config.yaml',\
                        tool_marker_config_file=tool_marker_dir+'weldgun_'+dataset_date+'_marker_config.yaml')
elif robot_type == 'R2':
    robot_marker_dir=config_dir+'MA1440_marker_config/'
    tool_marker_dir=config_dir+'mti_marker_config/'
    robot=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',\
                        tool_file_path=config_dir+'mti.csv',\
                        pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
                        base_marker_config_file=robot_marker_dir+'MA1440_'+dataset_date+'_marker_config.yaml',\
                        tool_marker_config_file=tool_marker_dir+'mti_'+dataset_date+'_marker_config.yaml')
jN = len(robot.robot.H[0])
robot.P_nominal=deepcopy(robot.robot.P)
robot.H_nominal=deepcopy(robot.robot.H)
robot.P_nominal=robot.P_nominal.T
robot.H_nominal=robot.H_nominal.T
robot = get_H_param_axis(robot)

#### using rigid body
use_toolmaker=True
T_base_basemarker = robot.T_base_basemarker
T_basemarker_base = T_base_basemarker.inv()

if use_toolmaker:
    robot.robot.R_tool = robot.T_toolmarker_flange.R
    robot.robot.p_tool = robot.T_toolmarker_flange.p
    robot.T_tool_toolmarker = Transform(np.eye(3),[0,0,0])

data_dir='PH_grad_data/test'+dataset_date+'_'+robot_type+'/train_data_'

robot_q = np.loadtxt(data_dir+'robot_q_align.csv',delimiter=',')
mocap_T = np.loadtxt(data_dir+'mocap_T_align.csv',delimiter=',')

assert len(robot_q)==len(mocap_T), f"Need to have the same amount of robot_q and mocap_T"

N_per_pose = 7
total_pose = int(len(robot_q)/N_per_pose)
robot_q_sample = deepcopy(robot_q[0:-1:N_per_pose])

train_N = total_pose
train_set=np.arange(train_N).astype(int)
print(data_dir)
print(train_set)

#### Gradient
plot_grad=False
plot_error=False
plot_block=False
save_PH = True
all_testing_pose=np.arange(N_per_pose)
# max_iteration = 500
# max_iteration = 1000
max_iteration = 200
# terminate_eps = 0.00005
terminate_eps = 0.0002
terminate_ori_error=999
# terminate_ori_error=0.05
P_size = 7
H_size = 6
alpha=0.1
# weight_ori = 1
# weight_pos = 1
weight_ori = 0.1
weight_pos = 1

# lambda_H = 5
# lambda_P = 0.5
lambda_H = 0.5
lambda_P = 0.5
start_t = time.time()

## get initial param from CPA
param_init = np.zeros(3*(jN+1)+2*jN)
param_init[:3*(jN+1)] = np.reshape(robot.calib_P.T,(3*(jN+1),))
for j in range(jN):
    sol = subproblem2(robot.H_nominal[j],robot.calib_H[:,j],robot.param_k2[j],robot.param_k1[j])
    sol_id = np.argmin(np.linalg.norm(sol,axis=1))
    sol_b = sol[sol_id][0]
    sol_a = sol[sol_id][1]
    # param_init[3*(jN+1)+2*j] = sol_a
    # param_init[3*(jN+1)+2*j+1] = sol_b
    param_init[3*(jN+1)+2*j] = np.degrees(sol_a)
    param_init[3*(jN+1)+2*j+1] = np.degrees(sol_b)

### start calibration. Iterate all collected configurations/clusters
PH_q = {}
for N in train_set:
    print("Training #"+str(N),"at Pose (q2q3):", np.round(np.degrees(robot_q_sample[N,1:3])))
    print(np.degrees(robot.robot.joint_lower_limit))
    print("Progress:",str(N)+"/"+str(total_pose),"Time Pass:",str(np.round(time.time()-start_t)))

    # initialize the parameters
    param = deepcopy(param_init)

    # start NLE (nonlinear esimation)
    st_iter = time.time()
    pos_error_progress = []
    pos_error_norm_progress = []
    ori_error_progress = []
    ori_error_norm_progress = []
    for iter_N in range(max_iteration):
        # update robot PH
        robot = get_PH_from_param(param,robot,unit='degrees')
        
        J_ana=[]
        error_pos_ori = []
        error_pos = []
        error_ori = []
        for testing_pose in all_testing_pose:
            pose_ind=N*N_per_pose+testing_pose
            J_ana_part = jacobian_param(param,robot,robot_q[pose_ind],unit='degrees')
            J_ana.extend(J_ana_part)
            # get error
            robot_init_T = robot.fwd(robot_q[pose_ind])
            T_marker_base = Transform(q2R(mocap_T[pose_ind][3:]),mocap_T[pose_ind][:3])
            T_tool_base = T_marker_base*robot.T_tool_toolmarker
            k,theta = R2rot(T_tool_base.R@robot_init_T.R.T)
            k=np.array(k)
            # error_pos_ori = np.append(error_pos_ori,np.append((k*theta)*weight_ori,(T_tool_base.p-robot_init_T.p)*weight_pos))
            error_pos_ori = np.append(error_pos_ori,np.append((k*np.degrees(theta))*weight_ori,(T_tool_base.p-robot_init_T.p)*weight_pos))
            error_pos.append(T_tool_base.p-robot_init_T.p)
            error_ori.append(k*np.degrees(theta))  # for plotting purpose only (unit: degrees)          
        J_ana = np.array(J_ana)
        pos_error_progress.append(error_pos[0])
        pos_error_norm_progress.append(np.linalg.norm(error_pos,ord=2,axis=1))
        ori_error_progress.append(error_ori[0])
        ori_error_norm_progress.append(np.linalg.norm(error_ori,ord=2,axis=1))
        
        if iter_N>0 and np.linalg.norm(pos_error_norm_progress[-1]-pos_error_norm_progress[-2])<terminate_eps and np.mean(ori_error_norm_progress[-1])<terminate_ori_error:
            break
        
        # update PH
        G = J_ana
        Kq = np.diag(np.append(np.ones(P_size*3)*lambda_P,np.ones(H_size*2))*lambda_H)
        H=np.matmul(G.T,G)+Kq
        H=(H+np.transpose(H))/2
        f=-np.matmul(G.T,error_pos_ori)
        dph=solve_qp(H,f,solver='quadprog')

        # alpha=fminbound(self.error_calc,0,0.999999999999999999999,args=(q_all[-1],qdot,curve_sliced_relative[i],))
        
        d_pH_update = alpha*dph
        param = param+d_pH_update

    print("Start/Final Mean Position Error:",round(np.mean(pos_error_norm_progress[0]),5),round(np.mean(pos_error_norm_progress[-1]),5))
    print("Start/Final Mean Orientation Error:",round(np.mean(ori_error_norm_progress[0]),5),round(np.mean(ori_error_norm_progress[-1]),5))
    print("Time iteration:",time.time()-st_iter)

    if plot_error:
        try:
            plt.close(fig)
        except:
            pass
        fig,axs = plt.subplots(2,3)
        axs[0,0].plot(np.array(pos_error_progress))
        axs[0,0].set_title("Position XYZ error of Pose 1")
        axs[0,1].plot(np.array(pos_error_norm_progress))
        axs[0,1].set_title("Position error norm of all poses")
        pos_error_diff = np.linalg.norm(np.diff(pos_error_norm_progress,axis=0),axis=1).flatten()
        axs[0,2].plot(np.array(pos_error_diff))
        axs[0,2].set_title("Position Error Norm Diff")
        axs[1,0].plot(np.array(ori_error_progress))
        axs[1,0].set_title("Orientation kdtheta error of Pose 1")
        axs[1,1].plot(np.array(ori_error_norm_progress))
        axs[1,1].set_title("Orientation error norm of all poses")
        ori_error_diff = np.linalg.norm(np.diff(ori_error_norm_progress,axis=0),axis=1).flatten()
        axs[1,2].plot(np.array(ori_error_diff))
        axs[1,2].set_title("Orientation Error Norm Diff")
        # fig.canvas.manager.window.wm_geometry("+%d+%d" % (1920+10,10))
        # fig.set_size_inches([13.95,7.92],forward=True)
        plt.tight_layout()
        plt.show(block=plot_block)
        # plt.pause(0.01)
        
        plt.errorbar(np.arange(len(pos_error_norm_progress)),np.mean(pos_error_norm_progress,axis=1),\
            yerr=np.mean(pos_error_norm_progress,axis=1))
        plt.xlabel('Iteration',fontsize=15)
        plt.xticks(np.arange(0,len(pos_error_norm_progress),len(pos_error_norm_progress)/6).astype(int),fontsize=15)
        plt.ylabel('Position Error Norm (mm)',fontsize=15)
        plt.yticks(fontsize=15)
        plt.title("Mean/Std of Position Error Norm of Poses",fontsize=18)
        plt.show()
        
        plt.errorbar(np.arange(len(ori_error_norm_progress)),np.mean(ori_error_norm_progress,axis=1),\
            yerr=np.mean(ori_error_norm_progress,axis=1))
        plt.xlabel('Iteration',fontsize=15)
        plt.xticks(np.arange(0,len(pos_error_norm_progress),len(pos_error_norm_progress)/6).astype(int),fontsize=15)
        plt.ylabel('Orientation Error Norm (deg)',fontsize=15)
        plt.yticks(fontsize=15)
        plt.title("Mean/Std of Orientation Error Norm of Poses",fontsize=18)
        plt.show()
    # update robot PH
    robot = get_PH_from_param(param,robot,unit='degrees')
    print(robot.robot.P.T)
    if save_PH:
        q_key = tuple(robot_q_sample[N,1:3])
        PH_q[q_key]={}
        PH_q[q_key]['P']=robot.robot.P
        PH_q[q_key]['H']=robot.robot.H
        PH_q[q_key]['train_pos_error']=pos_error_norm_progress
        PH_q[q_key]['train_ori_error']=ori_error_norm_progress
        with open(data_dir+'calib_PH_q_ana.pickle','wb') as file:
            pickle.dump(PH_q, file)

    print("================")
    
# exit()
###### get just one optimal pose
plot_grad=False
plot_error=False
alpha = 0.2
# initialize the parameters
param = np.zeros(3*(jN+1)+2*jN)
param[:3*(jN+1)] = np.reshape(robot.P_nominal,(3*(jN+1),))

# start NLE (nonlinear esimation)
pos_error_progress = []
pos_error_norm_progress = []
ori_error_progress = []
ori_error_norm_progress = []
for iter_N in range(max_iteration):
    if iter_N %50 == 0:
        print("================")
        print("One PH, iteration:",iter_N)
        if iter_N>0:
            print("Position error:",np.mean(pos_error_norm_progress[-1]))
            print("Orientation error:",np.mean(ori_error_norm_progress[-1]))

    # update robot PH
    robot = get_PH_from_param(param,robot)
    
    J_ana=[]
    error_pos_ori = []
    error_pos = []
    error_ori = []
    for pose_ind in range(len(robot_q)):
        J_ana_part = jacobian_param(param,robot,robot_q[pose_ind])
        J_ana.extend(J_ana_part)
        # get error
        robot_init_T = robot.fwd(robot_q[pose_ind])
        T_marker_base = Transform(q2R(mocap_T[pose_ind][3:]),mocap_T[pose_ind][:3])
        T_tool_base = T_marker_base*robot.T_tool_toolmarker
        k,theta = R2rot(T_tool_base.R@robot_init_T.R.T)
        k=np.array(k)
        error_pos_ori = np.append(error_pos_ori,np.append((k*theta)*weight_ori,(T_tool_base.p-robot_init_T.p)*weight_pos))
        error_pos.append(T_tool_base.p-robot_init_T.p)
        error_ori.append(k*np.degrees(theta))            
    J_ana = np.array(J_ana)
    pos_error_progress.append(error_pos[0])
    pos_error_norm_progress.append(np.linalg.norm(error_pos,ord=2,axis=1))
    ori_error_progress.append(error_ori[0])
    ori_error_norm_progress.append(np.linalg.norm(error_ori,ord=2,axis=1))
    
    if iter_N>0 and np.linalg.norm(pos_error_norm_progress[-1]-pos_error_norm_progress[-2])<terminate_eps:
        break

    # update PH
    G = J_ana
    Kq = np.diag(np.append(np.ones(P_size*3)*lambda_P,np.ones(H_size*2))*lambda_H)
    H=np.matmul(G.T,G)+Kq
    H=(H+np.transpose(H))/2
    f=-np.matmul(G.T,error_pos_ori)
    dph=solve_qp(H,f,solver='quadprog')

    d_pH_update = alpha*dph
    param = param+d_pH_update

robot = get_PH_from_param(param,robot)
print("Start/Final Mean Position Error:",round(np.mean(pos_error_norm_progress[0]),5),round(np.mean(pos_error_norm_progress[-1]),5))
print("Start/Final Mean Orientation Error:",round(np.mean(ori_error_norm_progress[0]),5),round(np.mean(ori_error_norm_progress[-1]),5))
print('P:',np.round(robot.robot.P,3).T)
print('H:',np.round(robot.robot.H,3).T)

if save_PH:
    PH_q={}
    PH_q['P']=robot.robot.P
    PH_q['H']=robot.robot.H
    PH_q['train_pos_error']=pos_error_norm_progress
    PH_q['train_ori_error']=ori_error_norm_progress
    with open(data_dir+'calib_one_PH_ana.pickle','wb') as file:
        pickle.dump(PH_q, file)

if plot_error:
    plt.clf()
    plt.plot(np.array(pos_error_progress))
    plt.title("Position XYZ error of Pose 1")
    plt.show()
    plt.plot(np.array(pos_error_norm_progress))
    plt.title("Position error norm of all poses")
    plt.show()
    error_diff = np.linalg.norm(np.diff(pos_error_norm_progress,axis=0),axis=1).flatten()
    plt.plot(error_diff)
    plt.title("Error Diff Norm")
    plt.show()    
    plt.plot(np.array(ori_error_progress))
    plt.title("Orientation kdtheta error of Pose 1")
    plt.show()
    plt.plot(np.array(ori_error_norm_progress))
    plt.title("Orientation error norm of all poses")
    plt.show()
plt.clf()
plt.plot(np.mean(pos_error_norm_progress,axis=1))
plt.title("Average Position error norm of all poses")
plt.show()

print("================")