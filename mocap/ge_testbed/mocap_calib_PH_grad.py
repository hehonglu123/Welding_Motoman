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

dataset_date='0913'

config_dir='config/'

robot_type = 'R1'

if robot_type=='R1':
    robot_name='M10ia'
    tool_name='ge_R1_tool'
    robot_marker_dir=config_dir+robot_name+'_marker_config/'
    tool_marker_dir=config_dir+tool_name+'_marker_config/'


elif robot_type=='R2':
    robot_name='LRMATE200id'
    tool_name='ge_R2_tool'
    robot_marker_dir=config_dir+robot_name+'_marker_config/'
    tool_marker_dir=config_dir+tool_name+'_marker_config/'
    

print("Dataset Date:",dataset_date)

robot=robot_obj(robot_name,def_path=config_dir+robot_name+'_robot_default_config.yml',tool_file_path=config_dir+tool_name+'.csv',\
base_marker_config_file=robot_marker_dir+robot_name+'_'+dataset_date+'_marker_config.yaml',tool_marker_config_file=tool_marker_dir+tool_name+'_'+dataset_date+'_marker_config.yaml')

#### using rigid body
use_toolmaker=True
T_base_basemarker = robot.T_base_basemarker
T_basemarker_base = T_base_basemarker.inv()

if use_toolmaker:
    robot.robot.R_tool = robot.T_toolmarker_flange.R
    robot.robot.p_tool = robot.T_toolmarker_flange.p
    robot.T_tool_toolmarker = Transform(np.eye(3),[0,0,0])

data_dir='PH_grad_data/test'+dataset_date+'_'+robot_type+'_part2/train_data_'

try:
    robot_q = np.loadtxt(data_dir+'robot_q_align.csv',delimiter=',')
    mocap_T = np.loadtxt(data_dir+'tool_T_align.csv',delimiter=',')
except:
    exit()

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
plot_error=True
plot_block=False
save_PH = False
all_testing_pose=np.arange(N_per_pose)
# max_iteration = 500
max_iteration = 200
terminate_eps = 0.0002
terminate_ori_error=999
# terminate_ori_error=0.05
total_grad_sample = 40
dP_up_range = 0.05
dP_low_range = 0.01
P_size = 7
dH_up_range = np.radians(0.1)
dH_low_range = np.radians(0.03)
H_size = 6
dH_rotate_axis = [[Rx,Ry],[Rz,Rx],[Rz,Rx],[Ry,Rz],[Rz,Rx],[Ry,Rz]]
alpha=0.5
# weight_ori = 0.1
weight_ori = 1
weight_pos = 1
lambda_H = 10
# lambda_H = 1
lambda_P = 1
start_t = time.time()

PH_q = {}
# train_set=[]
for N in train_set:
    print("Training #"+str(N),"at Pose (q2q3):", np.round(np.degrees(robot_q_sample[N,1:3])))
    print(np.degrees(robot.robot.joint_lower_limit))
    print("Progress:",str(N)+"/"+str(total_pose),"Time Pass:",str(np.round(time.time()-start_t)))

    pos_error_progress = []
    pos_error_norm_progress = []
    ori_error_progress = []
    ori_error_norm_progress = []
    robot_opt_P = deepcopy(robot.calib_P)
    robot_opt_H = deepcopy(robot.calib_H)
    for iter_N in range(max_iteration):
        # print(iter_N)

        d_T_all = [] # difference in robot T
        d_pH_all = [] # difference in PH
        for sample_N in range(total_grad_sample):
            # initial robot_T
            robot_init_P = deepcopy(robot_opt_P)
            robot_init_H = deepcopy(robot_opt_H)
            robot_pert_P = deepcopy(robot_opt_P)
            robot_pert_H = deepcopy(robot_opt_H)

            # change robot PH to perturb calib PH
            dP=rng.uniform(low=-(dP_up_range-dP_low_range),high=(dP_up_range-dP_low_range),size=robot_opt_P[:,:P_size].T.shape)
            dP=dP+dP/np.fabs(dP)*dP_low_range
            robot_pert_P[:,:P_size] = robot_pert_P[:,:P_size]+dP.T
            
            dH = np.zeros((H_size,2))
            for i in range(len(dH)):
                new_H = deepcopy(robot_pert_H[:,i])
                for j in range(2):
                    d_angle = rng.uniform(low=-(dH_up_range-dH_low_range),high=(dH_up_range-dH_low_range))
                    d_angle=d_angle+d_angle/np.fabs(d_angle)*dH_low_range
                    new_H = np.matmul(rot(dH_rotate_axis[i][j],d_angle),new_H)
                    dH[i,j]=np.degrees(d_angle)
                robot_pert_H[:,i] = deepcopy(new_H)
            d_pH_all.append(np.append(dP.flatten(),dH.flatten()))

            # iterate all surrounding poses
            for testing_pose in all_testing_pose:
                pose_ind=N*N_per_pose+testing_pose
                # print(np.round(np.degrees(robot_q[pose_ind])))
                robot.robot.P=deepcopy(robot_init_P)
                robot.robot.H=deepcopy(robot_init_H)
                robot_init_T = robot.fwd(robot_q[pose_ind])
                robot.robot.P=deepcopy(robot_pert_P)
                robot.robot.H=deepcopy(robot_pert_H)
                robot_pert_T = robot.fwd(robot_q[pose_ind])
                dR = robot_init_T.R.T@robot_pert_T.R
                k,theta = R2rot(dR)
                dPos = robot_pert_T.p-robot_init_T.p
                if theta == 0:
                    ktheta=np.zeros(3)
                else:
                    ktheta=k*np.degrees(theta)

                if testing_pose==all_testing_pose[0]:
                    d_T_all.append(np.append(dPos,ktheta))
                else:
                    d_T_all[sample_N] = np.append(d_T_all[sample_N],np.append(dPos,ktheta))
            # exit()
        #### get error
        error_pos_ori = []
        error_pos = []
        error_ori = []
        robot.robot.P=deepcopy(robot_init_P)
        robot.robot.H=deepcopy(robot_init_H)
        for testing_pose in all_testing_pose:
            pose_ind=N*N_per_pose+testing_pose
            robot_init_T = robot.fwd(robot_q[pose_ind])
            T_marker_base = Transform(q2R(mocap_T[pose_ind][3:]),mocap_T[pose_ind][:3])
            T_tool_base = T_marker_base*robot.T_tool_toolmarker
            k,theta = R2rot(robot_init_T.R.T@T_tool_base.R)
            k=np.array(k)
            error_pos_ori = np.append(error_pos_ori,np.append((T_tool_base.p-robot_init_T.p)*weight_pos,(k*np.degrees(theta))*weight_ori))
            error_pos.append(T_tool_base.p-robot_init_T.p)
            error_ori.append(k*np.degrees(theta))
        
        pos_error_progress.append(error_pos[0])
        pos_error_norm_progress.append(np.linalg.norm(error_pos,ord=2,axis=1))
        ori_error_progress.append(error_ori[0])
        ori_error_norm_progress.append(np.linalg.norm(error_ori,ord=2,axis=1))
        if iter_N>0 and np.linalg.norm(pos_error_norm_progress[-1]-pos_error_norm_progress[-2])<terminate_eps and np.mean(ori_error_norm_progress[-1])<terminate_ori_error:
            break

        # print(np.array(d_T_all).shape)
        # print(np.array(d_pH_all).shape)

        d_T_all = np.array(d_T_all).T
        d_pH_all = np.array(d_pH_all).T
        G = np.matmul(d_T_all,np.linalg.pinv(d_pH_all))
        
        if (iter_N==0) and plot_grad:
            plt.clf()
            print("Gradient Size:",G.shape)
            plt.matshow(G)
            plt.colorbar()
            plt.show(block=plot_block)
            
        
        # update PH
        Kq = np.diag(np.append(np.ones(P_size*3)*lambda_P,np.ones(H_size*2))*lambda_H)
        H=np.matmul(G.T,G)+Kq
        H=(H+np.transpose(H))/2
        f=-np.matmul(G.T,error_pos_ori)
        dph=solve_qp(H,f,solver='quadprog')

        # d_pH_update = alpha*np.matmul(np.linalg.pinv(G),error_pos_ori)
        d_pH_update = alpha*dph

        robot_opt_P[:,:P_size] = robot_opt_P[:,:P_size]+np.reshape(d_pH_update[:robot_opt_P[:,:P_size].size],robot_opt_P[:,:P_size].T.shape).T
        dH = np.reshape(d_pH_update[robot_opt_P[:,:P_size].size:],(H_size,2))
        for i in range(len(dH)):
            new_H = deepcopy(robot_opt_H[:,i])
            for j in range(2):
                d_angle = dH[i,j]
                new_H = np.matmul(rot(dH_rotate_axis[i][j],np.radians(d_angle)),new_H)
            robot_opt_H[:,i] = new_H

    print("Final Mean Position Error:",np.mean(pos_error_norm_progress[-1]))
    print("Final Mean Orientation Error:",np.mean(ori_error_norm_progress[-1]))
    print('P:',np.round(robot_opt_P,3).T)
    print('H:',np.round(robot_opt_H,3).T)

    if plot_error:
        try:
            plt.close(fig)
        except:
            pass
        # fig,axs = plt.subplots(2,3)
        # axs[0,0].plot(np.array(pos_error_progress))
        # axs[0,0].set_title("Position XYZ error of Pose 1")
        # axs[0,1].plot(np.array(pos_error_norm_progress))
        # axs[0,1].set_title("Position error norm of all poses")
        # pos_error_diff = np.linalg.norm(np.diff(pos_error_norm_progress,axis=0),axis=1).flatten()
        # axs[0,2].plot(np.array(pos_error_diff))
        # axs[0,2].set_title("Position Error Norm Diff")
        # axs[1,0].plot(np.array(ori_error_progress))
        # axs[1,0].set_title("Orientation kdtheta error of Pose 1")
        # axs[1,1].plot(np.array(ori_error_norm_progress))
        # axs[1,1].set_title("Orientation error norm of all poses")
        # ori_error_diff = np.linalg.norm(np.diff(ori_error_norm_progress,axis=0),axis=1).flatten()
        # axs[1,2].plot(np.array(ori_error_diff))
        # axs[1,2].set_title("Orientation Error Norm Diff")
        # # fig.canvas.manager.window.wm_geometry("+%d+%d" % (1920+10,10))
        # # fig.set_size_inches([13.95,7.92],forward=True)
        # plt.tight_layout()
        # plt.show(block=plot_block)
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
    
    if save_PH:
        q_key = tuple(robot_q_sample[N,1:3])
        PH_q[q_key]={}
        PH_q[q_key]['P']=robot_opt_P
        PH_q[q_key]['H']=robot_opt_H
        PH_q[q_key]['train_pos_error']=pos_error_norm_progress
        PH_q[q_key]['train_ori_error']=ori_error_norm_progress
        with open(data_dir+'calib_PH_q.pickle','wb') as file:
            pickle.dump(PH_q, file)

    print("================")

# get just one optimal pose
plot_grad=False
plot_error=False
pos_error_progress = []
pos_error_norm_progress = []
ori_error_progress = []
ori_error_norm_progress = []
robot_opt_P = deepcopy(robot.calib_P)
robot_opt_H = deepcopy(robot.calib_H)
for iter_N in range(max_iteration):
    print(iter_N)

    d_T_all = [] # difference in robot T
    d_pH_all = [] # difference in PH
    for sample_N in range(total_grad_sample):
        # initial robot_T
        robot_init_P = deepcopy(robot_opt_P)
        robot_init_H = deepcopy(robot_opt_H)
        robot_pert_P = deepcopy(robot_opt_P)
        robot_pert_H = deepcopy(robot_opt_H)

        # change robot PH to perturb calib PH
        dP=rng.uniform(low=-(dP_up_range-dP_low_range),high=(dP_up_range-dP_low_range),size=robot_opt_P[:,:P_size].T.shape)
        dP=dP+dP/np.fabs(dP)*dP_low_range
        robot_pert_P[:,:P_size] = robot_pert_P[:,:P_size]+dP.T
        
        dH = np.zeros((H_size,2))
        for i in range(len(dH)):
            new_H = deepcopy(robot_pert_H[:,i])
            for j in range(2):
                d_angle = rng.uniform(low=-(dH_up_range-dH_low_range),high=(dH_up_range-dH_low_range))
                d_angle=d_angle+d_angle/np.fabs(d_angle)*dH_low_range
                new_H = np.matmul(rot(dH_rotate_axis[i][j],d_angle),new_H)
                dH[i,j]=np.degrees(d_angle)
            robot_pert_H[:,i] = deepcopy(new_H)
        d_pH_all.append(np.append(dP.flatten(),dH.flatten()))

        # iterate all surrounding poses
        for pose_ind in range(len(robot_q)):
            # print(np.round(np.degrees(robot_q[pose_ind])))
            robot.robot.P=deepcopy(robot_init_P)
            robot.robot.H=deepcopy(robot_init_H)
            robot_init_T = robot.fwd(robot_q[pose_ind])
            robot.robot.P=deepcopy(robot_pert_P)
            robot.robot.H=deepcopy(robot_pert_H)
            robot_pert_T = robot.fwd(robot_q[pose_ind])
            dR = robot_init_T.R.T@robot_pert_T.R
            k,theta = R2rot(dR)
            dPos = robot_pert_T.p-robot_init_T.p
            if theta == 0:
                ktheta=np.zeros(3)
            else:
                ktheta=k*np.degrees(theta)

            if pose_ind==0:
                d_T_all.append(np.append(dPos,ktheta))
            else:
                d_T_all[sample_N] = np.append(d_T_all[sample_N],np.append(dPos,ktheta))
        # exit()
    #### get error
    error_pos_ori = []
    error_pos = []
    error_ori = []
    robot.robot.P=deepcopy(robot_init_P)
    robot.robot.H=deepcopy(robot_init_H)
    for pose_ind in range(len(robot_q)):
        robot_init_T = robot.fwd(robot_q[pose_ind])
        T_marker_base = Transform(q2R(mocap_T[pose_ind][3:]),mocap_T[pose_ind][:3])
        T_tool_base = T_marker_base*robot.T_tool_toolmarker
        k,theta = R2rot(robot_init_T.R.T@T_tool_base.R)
        k=np.array(k)
        error_pos_ori = np.append(error_pos_ori,np.append((T_tool_base.p-robot_init_T.p)*weight_pos,(k*np.degrees(theta))*weight_ori))
        error_pos.append(T_tool_base.p-robot_init_T.p)
        error_ori.append(k*np.degrees(theta))
    
    pos_error_progress.append(error_pos[0])
    pos_error_norm_progress.append(np.linalg.norm(error_pos,ord=2,axis=1))
    ori_error_progress.append(error_ori[0])
    ori_error_norm_progress.append(np.linalg.norm(error_ori,ord=2,axis=1))
    if iter_N>0 and np.linalg.norm(pos_error_norm_progress[-1]-pos_error_norm_progress[-2])<terminate_eps:
        break

    # print(np.array(d_T_all).shape)
    # print(np.array(d_pH_all).shape)

    d_T_all = np.array(d_T_all).T
    d_pH_all = np.array(d_pH_all).T
    G = np.matmul(d_T_all,np.linalg.pinv(d_pH_all))
    
    if (iter_N==0) and plot_grad:
        plt.clf()
        print("Gradient Size:",G.shape)
        plt.matshow(G)
        plt.colorbar()
        plt.show(block=plot_block)
    
    # update PH
    Kq = np.diag(np.append(np.ones(P_size*3)*lambda_P,np.ones(H_size*2))*lambda_H)
    H=np.matmul(G.T,G)+Kq
    H=(H+np.transpose(H))/2
    f=-np.matmul(G.T,error_pos_ori)
    dph=solve_qp(H,f,solver='quadprog')

    # d_pH_update = alpha*np.matmul(np.linalg.pinv(G),error_pos_ori)
    d_pH_update = alpha*dph

    robot_opt_P[:,:P_size] = robot_opt_P[:,:P_size]+np.reshape(d_pH_update[:robot_opt_P[:,:P_size].size],robot_opt_P[:,:P_size].T.shape).T
    dH = np.reshape(d_pH_update[robot_opt_P[:,:P_size].size:],(H_size,2))
    for i in range(len(dH)):
        new_H = deepcopy(robot_opt_H[:,i])
        for j in range(2):
            d_angle = dH[i,j]
            new_H = np.matmul(rot(dH_rotate_axis[i][j],np.radians(d_angle)),new_H)
        robot_opt_H[:,i] = new_H

print("Final Mean Position Error:",np.mean(pos_error_norm_progress[-1]))
print('P:',np.round(robot_opt_P,3).T)
print('H:',np.round(robot_opt_H,3).T)

if save_PH:
    PH_q={}
    PH_q['P']=robot_opt_P
    PH_q['H']=robot_opt_H
    PH_q['train_pos_error']=pos_error_norm_progress
    PH_q['train_ori_error']=ori_error_norm_progress
    with open(data_dir+'calib_one_PH.pickle','wb') as file:
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