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

dataset_date='0801'

config_dir='../config/'

robot_type = 'R1'

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

#### using rigid body
use_toolmaker=True
T_base_basemarker = robot.T_base_basemarker
T_basemarker_base = T_base_basemarker.inv()

if use_toolmaker:
    robot.robot.R_tool = robot.T_toolmarker_flange.R
    robot.robot.p_tool = robot.T_toolmarker_flange.p
    robot.T_tool_toolmarker = Transform(np.eye(3),[0,0,0])

data_dir='PH_grad_data/test'+dataset_date+'_'+robot_type+'/train_data_'

try:
    use_raw=False
    
    robot_q = np.loadtxt(data_dir+'robot_q_align.csv',delimiter=',')
    mocap_T = np.loadtxt(data_dir+'mocap_T_align.csv',delimiter=',')
    
    if use_raw:
        mocap_T=[]
        toolrigid_raw = np.loadtxt(data_dir+'tool_T_raw.csv',delimiter=',')
        baserigid_raw = np.loadtxt(data_dir+'base_T_raw.csv',delimiter=',')
        
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
                mocap_T.append(this_T)
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
                mocap_T.append(this_T)
                pos_toolrigid=[]
                break

except:

    with open(data_dir+'robot_q_cont.pickle', 'rb') as handle:
        robot_q = pickle.load(handle)
        robot_q = robot_q[:,:6]
    with open(data_dir+'robot_timestamps_cont.pickle', 'rb') as handle:
        robot_q_stamps = pickle.load(handle)
    robot_qdot = np.divide(np.gradient(robot_q,axis=0),np.tile(np.gradient(robot_q_stamps),(6,1)).T)
    robot_qdot_norm = np.linalg.norm(robot_qdot,axis=1)

    marker_id = robot.tool_rigid_id
    with open(data_dir+'mocap_p_cont.pickle', 'rb') as handle:
        mocap_p = pickle.load(handle)
        # static_mocap_marker = mocap_p[robot.base_markers_id[0]]
        base_rigid_p = mocap_p[robot.base_rigid_id]
        mocap_p = np.array(mocap_p[marker_id])
    with open(data_dir+'mocap_R_cont.pickle', 'rb') as handle:
        mocap_R = pickle.load(handle)
        # static_mocap_marker = mocap_p[robot.base_markers_id[0]]
        base_rigid_R = mocap_R[robot.base_rigid_id]
        mocap_R = np.array(mocap_R[marker_id])
    with open(data_dir+'mocap_timestamps_cont.pickle', 'rb') as handle:
        mocap_stamps = pickle.load(handle)
        base_rigid_stamps = np.array(mocap_stamps[marker_id])
        mocap_stamps = np.array(mocap_stamps[marker_id])
    mocap_pdot = np.divide(np.gradient(mocap_p,axis=0),np.tile(np.gradient(mocap_stamps),(3,1)).T)
    mocap_pdot_norm = np.linalg.norm(mocap_pdot,axis=1)

    print(len(mocap_p))
    print(len(mocap_stamps))
    print(len(base_rigid_p))

    mocap_start_k = 1400
    mocap_end_k = -110
    mocap_R = mocap_R[mocap_start_k:mocap_end_k]
    mocap_p = mocap_p[mocap_start_k:mocap_end_k]
    mocap_stamps = mocap_stamps[mocap_start_k:mocap_end_k]
    mocap_pdot = mocap_pdot[mocap_start_k:mocap_end_k]
    mocap_pdot_norm = mocap_pdot_norm[mocap_start_k:mocap_end_k]
    base_rigid_p=base_rigid_p[mocap_start_k:mocap_end_k]
    base_rigid_R=base_rigid_R[mocap_start_k:mocap_end_k]
    base_rigid_stamps=base_rigid_stamps[mocap_start_k:mocap_end_k]

    plt.plot(robot_qdot_norm)
    plt.show()
    plt.plot(mocap_pdot_norm)
    plt.show()
    
    timewindow = 0.3

    # the robot stop 3 sec, 
    # find "2 sec window" with smallest norm of velocity deviation
    robot_vdev_thres = 0.005
    robot_v_thres = 0.03
    dt_ave_windown = 1000
    robot_v_dev = []
    robot_stop_k = []
    v_dev_flag = False
    v_dev_local = []
    dK_robot = int(timewindow/np.mean(np.gradient(robot_q_stamps)))
    all_dkrobot=[]
    for i in range(0,len(robot_q)-dK_robot):
        dt_lookahead = min([i+dt_ave_windown,len(robot_q)])
        dK_robot = int(timewindow/np.mean(np.gradient(robot_q_stamps[i:dt_lookahead]))) 
        robot_v_dev.append(np.std(robot_qdot_norm[i:i+dK_robot]))
        if robot_v_dev[-1]<robot_vdev_thres and robot_qdot_norm[i]<robot_v_thres:
            v_dev_flag=True
            v_dev_local.append(robot_v_dev[-1])
        else:
            if v_dev_flag:
                v_dev_flag=False
                local_argmin = np.argmin(v_dev_local)
                robot_stop_k.append(i-len(v_dev_local)+local_argmin)
                all_dkrobot.append(dK_robot)
                v_dev_local=[]
    robot_v_dev=np.array(robot_v_dev)
    robot_stop_k.append(np.argmin(robot_v_dev[robot_stop_k[-1]+dK_robot:])+robot_stop_k[-1]+dK_robot)
    all_dkrobot.append(dK_robot)

    mocap_vdev_thres = 10
    mocap_v_thres = 50
    dt_ave_mocap = np.mean(np.gradient(mocap_stamps))
    dK_mocap = int(timewindow/dt_ave_mocap)
    print(dt_ave_mocap)
    print(dK_mocap)
    mocap_v_dev = []
    mocap_stop_k = []
    v_dev_flag = False
    v_dev_local = []
    for i in range(0,len(mocap_pdot)-dK_mocap):
        mocap_v_dev.append(np.std(mocap_pdot_norm[i:i+dK_mocap]))
        if mocap_v_dev[-1]<mocap_vdev_thres and mocap_pdot_norm[i]<mocap_v_thres:
            v_dev_flag=True
            v_dev_local.append(mocap_v_dev[-1])
        else:
            if v_dev_flag:
                v_dev_flag=False
                local_argmin = np.argmin(v_dev_local)
                mocap_stop_k.append(i-len(v_dev_local)+local_argmin)
                v_dev_local=[]
    mocap_v_dev = np.array(mocap_v_dev)
    mocap_stop_k.append(np.argmin(mocap_v_dev[mocap_stop_k[-1]+dK_mocap:])+mocap_stop_k[-1]+dK_mocap)

    # check 
    plt.plot(robot_v_dev)
    plt.scatter(robot_stop_k,robot_v_dev[robot_stop_k])
    plt.plot(robot_qdot_norm,'blue')
    for ki in range(len(robot_stop_k)):
        k=robot_stop_k[ki]
        dK_robot=all_dkrobot[ki]
        plt.plot(np.arange(k,k+dK_robot),robot_qdot_norm[k:k+dK_robot])
    plt.show()
    plt.plot(mocap_v_dev)
    plt.plot(mocap_pdot_norm,'blue')
    for k in mocap_stop_k:
        plt.plot(np.arange(k,k+dK_mocap),mocap_pdot_norm[k:k+dK_mocap])
    plt.show()

    # check total stop
    print("Total robot stop:",len(robot_stop_k))
    print("Total mocap stop:",len(mocap_stop_k))

    # assert len(robot_stop_k)==len(mocap_stop_k), f"Mocap Stop and Robot Stop should be the same."

    if len(robot_stop_k)!=len(mocap_stop_k):
        # change robot PH to calib PH
        robot.robot.P = robot.calib_P
        robot.robot.H = robot.calib_H
        # change to calibrated flange (tool rigidbody orientation)
        robot.robot.T_flange = robot.T_tool_flange

        threshold = 3
        erase_robot_stop = []
        mocap_start_offset = 0
        for i in range(len(robot_stop_k)):
            this_robot_q = np.mean(robot_q[robot_stop_k[i]:robot_stop_k[i]+all_dkrobot[i]],axis=0)
            rob_T = robot.fwd(this_robot_q)

            find_flag = False
            for j in range(i-mocap_start_offset,i-mocap_start_offset+5):
                if j>=len(mocap_stop_k):
                    break
                T_mocap_basemarker = Transform(q2R(base_rigid_R[mocap_stop_k[j]]),base_rigid_p[mocap_stop_k[j]]).inv()
                T_marker_mocap = Transform(q2R(mocap_R[mocap_stop_k[j]]),mocap_p[mocap_stop_k[j]])
                T_marker_base = T_basemarker_base*T_mocap_basemarker*T_marker_mocap
                # print(rob_T)
                # print(T_marker_base)
                # print(np.linalg.norm(rob_T.p-T_marker_base.p))
                # exit()
                if np.round(np.linalg.norm(rob_T.p-T_marker_base.p))<threshold:
                    find_flag=True
                    break
            if not find_flag:
                erase_robot_stop.append(i)
                mocap_start_offset+=1
                print(erase_robot_stop)
        robot_stop_k = np.delete(robot_stop_k,erase_robot_stop)
        all_dkrobot = np.delete(all_dkrobot,erase_robot_stop)
        
        erase_both_stop=[]
        robot_stop_k_groups=[]
        all_dkrobot_groups=[]
        mocap_stop_k_groups=[]
        threshold = 1
        i = 0
        while True:
            this_robot_q = np.mean(robot_q[robot_stop_k[i]:robot_stop_k[i]+all_dkrobot[i]],axis=0)
            find_flag = False
            for j in range(i+1,i+7):
                if j>=len(robot_stop_k):
                    find_flag=True
                    break
                compare_q = np.mean(robot_q[robot_stop_k[j]:robot_stop_k[j]+all_dkrobot[j]],axis=0)
                if np.linalg.norm(np.degrees(this_robot_q[1:3]-compare_q[1:3]))>threshold:
                    find_flag=True
                    break
            if not find_flag:
                robot_stop_k_groups.extend(robot_stop_k[i:i+7])
                all_dkrobot_groups.extend(all_dkrobot[i:i+7])
                mocap_stop_k_groups.extend(mocap_stop_k[i:i+7])
                i+=7
            else:
                i=j
            
            if i>=len(robot_stop_k):
                break

        robot_stop_k = robot_stop_k_groups
        all_dkrobot = all_dkrobot_groups
        mocap_stop_k = mocap_stop_k_groups

    assert len(robot_stop_k)==len(mocap_stop_k), f"Mocap Stop and Robot Stop should be the same."

    robot_stop_q = []
    mocap_stop_T = []
    for i in range(len(robot_stop_k)):
        this_robot_q = np.mean(robot_q[robot_stop_k[i]:robot_stop_k[i]+all_dkrobot[i]],axis=0)
        this_mocap_ori = []
        this_mocap_p = []
        for k in range(mocap_stop_k[i],min(mocap_stop_k[i]+dK_mocap,len(base_rigid_R))):
            T_mocap_basemarker = Transform(q2R(base_rigid_R[k]),base_rigid_p[k]).inv()
            T_marker_mocap = Transform(q2R(mocap_R[k]),mocap_p[k])
            T_marker_base = T_basemarker_base*T_mocap_basemarker*T_marker_mocap
            this_mocap_ori.append(R2rpy(T_marker_base.R))
            this_mocap_p.append(T_marker_base.p)
        this_mocap_p = np.mean(this_mocap_p,axis=0)
        this_mocap_ori = R2q(rpy2R(np.mean(this_mocap_ori,axis=0)))
        robot_stop_q.append(this_robot_q)
        mocap_stop_T.append(np.append(this_mocap_p,this_mocap_ori))
    robot_q = robot_stop_q
    mocap_T = mocap_stop_T

    np.savetxt(data_dir+'robot_q_align.csv',robot_q,delimiter=',')
    np.savetxt(data_dir+'mocap_T_align.csv',mocap_T,delimiter=',')

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