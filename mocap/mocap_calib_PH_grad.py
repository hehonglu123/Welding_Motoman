import numpy as np
from copy import deepcopy
from general_robotics_toolbox import *
import pickle
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
base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

T_base_basemarker = robot_weld.T_base_basemarker
T_basemarker_base = T_base_basemarker.inv()
robot_weld.T_tool_toolmarker=Transform(np.eye(3),[0,0,0])
robot_weld.robot.T_flange = robot_weld.T_tool_flange

data_dir='PH_grad_data/test0516_R1/train_data_'

try:
    robot_q = np.loadtxt(data_dir+'robot_q_align.csv',delimiter=',')
    mocap_T = np.loadtxt(data_dir+'mocap_T_align.csv',delimiter=',')

except:

    with open(data_dir+'robot_q_cont.pickle', 'rb') as handle:
        robot_q = pickle.load(handle)
        robot_q = robot_q[:,:6]
    with open(data_dir+'robot_timestamps_cont.pickle', 'rb') as handle:
        robot_q_stamps = pickle.load(handle)
    robot_qdot = np.divide(np.gradient(robot_q,axis=0),np.tile(np.gradient(robot_q_stamps),(6,1)).T)
    robot_qdot_norm = np.linalg.norm(robot_qdot,axis=1)

    marker_id = robot_weld.tool_rigid_id
    with open(data_dir+'mocap_p_cont.pickle', 'rb') as handle:
        mocap_p = pickle.load(handle)
        # static_mocap_marker = mocap_p[robot_weld.base_markers_id[0]]
        base_rigid_p = mocap_p[robot_weld.base_rigid_id]
        mocap_p = np.array(mocap_p[marker_id])
    with open(data_dir+'mocap_R_cont.pickle', 'rb') as handle:
        mocap_R = pickle.load(handle)
        # static_mocap_marker = mocap_p[robot_weld.base_markers_id[0]]
        base_rigid_R = mocap_R[robot_weld.base_rigid_id]
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

    mocap_start_k = 1200
    mocap_R = mocap_R[mocap_start_k:]
    mocap_p = mocap_p[mocap_start_k:]
    mocap_stamps = mocap_stamps[mocap_start_k:]
    mocap_pdot = mocap_pdot[mocap_start_k:]
    mocap_pdot_norm = mocap_pdot_norm[mocap_start_k:]
    base_rigid_p=base_rigid_p[mocap_start_k:]
    base_rigid_R=base_rigid_R[mocap_start_k:]
    base_rigid_stamps=base_rigid_stamps[mocap_start_k:]

    plt.plot(mocap_pdot_norm)
    plt.show()
    plt.plot(robot_qdot_norm)
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
        robot_weld.robot.P = robot_weld.calib_P
        robot_weld.robot.H = robot_weld.calib_H
        # change to calibrated flange (tool rigidbody orientation)
        robot_weld.robot.T_flange = robot_weld.T_tool_flange

        threshold = 3
        erase_robot_stop = []
        mocap_start_offset = 0
        for i in range(len(robot_stop_k)):
            this_robot_q = np.mean(robot_q[robot_stop_k[i]:robot_stop_k[i]+all_dkrobot[i]],axis=0)
            rob_T = robot_weld.fwd(this_robot_q)

            find_flag = False
            for j in range(i-mocap_start_offset,i-mocap_start_offset+5):
                T_mocap_basemarker = Transform(q2R(base_rigid_R[mocap_stop_k[j]]),base_rigid_p[mocap_stop_k[j]]).inv()
                T_marker_mocap = Transform(q2R(mocap_R[mocap_stop_k[j]]),mocap_p[mocap_stop_k[j]])
                T_marker_base = T_basemarker_base*T_mocap_basemarker*T_marker_mocap
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
        threshold = 1
        for i in range(0,len(robot_stop_k),7):
            this_robot_q = np.mean(robot_q[robot_stop_k[i]:robot_stop_k[i]+all_dkrobot[i]],axis=0)
            find_flag = False
            for j in range(i+1,i+7):
                compare_q = np.mean(robot_q[robot_stop_k[j]:robot_stop_k[j]+all_dkrobot[j]],axis=0)
                if np.linalg.norm(np.degrees(this_robot_q[1:3]-compare_q[1:3]))>threshold:
                    find_flag=True
                    break
            if find_flag:
                erase_both_stop = np.arange(i,i+7-len(erase_robot_stop))
                print(erase_both_stop)
                break
        robot_stop_k = np.delete(robot_stop_k,erase_both_stop)
        all_dkrobot = np.delete(all_dkrobot,erase_both_stop)
        mocap_stop_k = np.delete(mocap_stop_k,erase_both_stop)

    robot_stop_q = []
    mocap_stop_T = []
    for i in range(len(robot_stop_k)):
        this_robot_q = np.mean(robot_q[robot_stop_k[i]:robot_stop_k[i]+all_dkrobot[i]],axis=0)
        this_mocap_ori = []
        this_mocap_p = []
        for k in range(mocap_stop_k[i],mocap_stop_k[i]+dK_mocap):
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

train_N = 3
# choosing poses
robot_q_sample = deepcopy(robot_q[0:-1:N_per_pose])
train_set=[]
# linearly choose poses
set1desired = np.linspace(robot_q_sample[0][1:3],np.zeros(2),int(train_N/2)+1)
set2desired = np.linspace(np.zeros(2),robot_q_sample[-1][1:3],int(train_N/2)+1)
setdesired = np.vstack((set1desired,set2desired[1:]))
for desired_q in setdesired:
    q_index = np.argmin(np.linalg.norm(robot_q_sample[:,1:3]-desired_q,ord=2,axis=1))
    train_set.append(q_index)

#### Gradient
plot_grad=True
all_testing_pose=np.arange(N_per_pose)
total_iteration = 100
total_grad_sample = 80
# total_grad_sample = 7
dP_up_range = 0.05
dP_low_range = 0.01
P_size = 6
dH_up_range = np.radians(0.1)
dH_low_range = np.radians(0.03)
H_size = 6
dH_rotate_axis = [[Rx,Ry],[Rz,Rx],[Rz,Rx],[Ry,Rz],[Rz,Rx],[Ry,Rz]]
alpha=0.1
weight_ori = 0.01
weight_pos = 1
for N in train_set:
    print("Training at Pose (q2q3):", np.round(np.degrees(robot_q_sample[N,1:3])))

    pos_error_progress = []
    pos_error_norm_progress = []
    ori_error_progress = []
    ori_error_norm_progress = []
    robot_opt_P = deepcopy(robot_weld.calib_P)
    robot_opt_H = deepcopy(robot_weld.calib_H)
    for iter_N in range(total_iteration):
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
            for testing_pose in all_testing_pose:
                pose_ind=N*N_per_pose+testing_pose
                # print(np.round(np.degrees(robot_q[pose_ind])))
                robot_weld.robot.P=deepcopy(robot_init_P)
                robot_weld.robot.H=deepcopy(robot_init_H)
                robot_init_T = robot_weld.fwd(robot_q[pose_ind])
                robot_weld.robot.P=deepcopy(robot_pert_P)
                robot_weld.robot.H=deepcopy(robot_pert_H)
                robot_pert_T = robot_weld.fwd(robot_q[pose_ind])
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
        # get error
        error_pos_ori = []
        error_pos = []
        error_ori = []
        robot_weld.robot.P=deepcopy(robot_init_P)
        robot_weld.robot.H=deepcopy(robot_init_H)
        for testing_pose in all_testing_pose:
            pose_ind=N*N_per_pose+testing_pose
            robot_init_T = robot_weld.fwd(robot_q[pose_ind])
            T_marker_base = Transform(q2R(mocap_T[pose_ind][3:]),mocap_T[pose_ind][:3])
            T_tool_base = T_marker_base*robot_weld.T_tool_toolmarker
            k,theta = R2rot(robot_init_T.R.T@T_tool_base.R)
            k=np.array(k)
            error_pos_ori = np.append(error_pos_ori,np.append((T_tool_base.p-robot_init_T.p)*weight_pos,(k*np.degrees(theta))*weight_ori))
            error_pos.append(T_tool_base.p-robot_init_T.p)
            error_ori.append(k*np.degrees(theta))

        # print(np.array(d_T_all).shape)
        # print(np.array(d_pH_all).shape)

        d_T_all = np.array(d_T_all).T
        d_pH_all = np.array(d_pH_all).T
        G = np.matmul(d_T_all,np.linalg.pinv(d_pH_all))
        
        if (iter_N==0) and plot_grad:
            print("Gradient Size:",G.shape)
            plt.matshow(G)
            plt.colorbar()
            plt.show()
        
        # update PH
        Kq = np.diag(np.append(np.ones(P_size*3),np.ones(H_size*2))*10)
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

        pos_error_progress.append(error_pos[0])
        pos_error_norm_progress.append(np.linalg.norm(error_pos,ord=2,axis=1))
        ori_error_progress.append(error_ori[0])
        ori_error_norm_progress.append(np.linalg.norm(error_ori,ord=2,axis=1))

    plt.plot(np.array(pos_error_progress))
    plt.title("Position XYZ error of Pose 1")
    plt.show()
    plt.plot(np.array(pos_error_norm_progress))
    plt.title("Position error norm of all poses")
    plt.show()
    plt.plot(np.array(ori_error_progress))
    plt.title("Orientation kdtheta error of Pose 1")
    plt.show()
    plt.plot(np.array(ori_error_norm_progress))
    plt.title("Orientation error norm of all poses")
    plt.show()

exit()

# change robot PH to calib PH
robot_weld.robot.P = robot_weld.calib_P
robot_weld.robot.H = robot_weld.calib_H
# change to calibrated flange (tool rigidbody orientation)
robot_weld.robot.T_flange = robot_weld.T_tool_flange
# from tool rigid to tool tip
robot_weld.robot.R_tool=robot_weld.T_tool_toolmarker.R
robot_weld.robot.p_tool=robot_weld.T_tool_toolmarker.p
robot_weld.p_tool=robot_weld.T_tool_toolmarker.p
robot_weld.R_tool=robot_weld.T_tool_toolmarker.R

# final PH
print("Final PH")
print("P:",robot_weld.robot.P[:,1:].T)
print("H:",robot_weld.robot.H.T)

# on training set
for testing_pose in all_testing_pose:
    position_error=[]
    orientation_error=[]
    mocap_position=[]
    mocap_orientation=[]
    robt_position = []
    std_pos_N = []
    std_pos_norm_N = []
    std_ori_N = []
    std_ori_norm_N = []
    for N in train_N:
        robot_k = robot_stop_k[N*total_pose+testing_pose]
        mocap_k = mocap_stop_k[N*total_pose+testing_pose]
        robot_T = robot_weld.fwd(robot_q[robot_k])
        
        this_mocap_p = mocap_p[mocap_k:mocap_k+dK_mocap]
        # convert to basemarker than base
        T_mocap_basemarker = Transform(base_rigid_R[mocap_k],base_rigid_p[mocap_k]).inv()
        this_mocap_p = []
        this_mocap_R = []
        this_orientation_error = []
        for k in range(mocap_k,mocap_k+dK_mocap):
            T_marker_mocap = Transform(mocap_R[k],mocap_p[k])
            T_marker_base = T_basemarker_base*T_mocap_basemarker*T_marker_mocap
            T_tool_base = T_marker_base*robot_weld.T_tool_toolmarker
            this_mocap_p.append(T_tool_base.p)
            this_mocap_R.append(np.degrees(R2rpy(mocap_R[k])))
            this_orientation_error.append(np.degrees(R2rpy(robot_T.R.T@T_tool_base.R)))

        mocap_position.append(np.mean(this_mocap_p,axis=0))
        robt_position.append(robot_T.p)
        position_error.append(np.mean(this_mocap_p,axis=0)-robot_T.p)
        std_pos_N.append(np.std(this_mocap_p,axis=0))
        std_pos_norm_N.append(np.std(np.linalg.norm(this_mocap_p,2,axis=1)))
        
        orientation_error.append(np.mean(this_orientation_error,axis=0))
        std_ori_N.append(np.std(this_mocap_R,axis=0))
        std_ori_norm_N.append(np.std(np.linalg.norm(this_mocap_R,2,axis=1)))

    print("Training Set on Pose",testing_pose)
    print("Mean Position:",np.mean(mocap_position,axis=0))
    print("Mean FK Position:",np.mean(robt_position,axis=0))
    print("Mean Position Error Vec:",np.mean(position_error,axis=0))
    print("Mean Abs Position Error Vec:",np.mean(np.fabs(position_error),axis=0))
    print("Mean Position Error:",np.mean(np.linalg.norm(position_error,2,axis=1)))
    print("Std Position:",np.std(mocap_position,axis=0))
    print("Std FK Position:",np.std(robt_position,axis=0))
    print("Std Position Error:",np.std(position_error,axis=0))
    print("Std Position Error Norm:",np.std(np.linalg.norm(position_error,2,axis=1)))
    print("Mean N Std Position:",np.mean(std_pos_N,axis=0))
    print("Mean N Std Position Norm:",np.mean(std_pos_norm_N,axis=0))
    print("===")
    print("Mean Orientation Error Vec:",np.mean(orientation_error,axis=0))
    print("Mean Orientation Error:",np.mean(np.linalg.norm(orientation_error,2,axis=1)))
    print("Std Orientation Error:",np.std(orientation_error,axis=0))
    print("Std Orientation Error Norm:",np.std(np.linalg.norm(orientation_error,2,axis=1)))
    print("Mean N Std Orientation:",np.mean(std_ori_N,axis=0))
    print("Mean N Std Orientation Norm:",np.mean(std_ori_norm_N,axis=0))
    print("===========================")

# on testing set
for testing_pose in all_testing_pose:
    position_error=[]
    orientation_error=[]
    mocap_position=[]
    mocap_orientation=[]
    robt_position = []
    std_pos_N = []
    std_pos_norm_N = []
    std_ori_N = []
    std_ori_norm_N = []
    for N in test_N:
        robot_k = robot_stop_k[N*total_pose+testing_pose]
        mocap_k = mocap_stop_k[N*total_pose+testing_pose]
        robot_T = robot_weld.fwd(robot_q[robot_k])
        
        this_mocap_p = mocap_p[mocap_k:mocap_k+dK_mocap]
        # convert to basemarker than base
        T_mocap_basemarker = Transform(base_rigid_R[mocap_k],base_rigid_p[mocap_k]).inv()
        this_mocap_p = []
        this_mocap_R = []
        this_orientation_error = []
        for k in range(mocap_k,mocap_k+dK_mocap):
            T_marker_mocap = Transform(mocap_R[k],mocap_p[k])
            T_marker_base = T_basemarker_base*T_mocap_basemarker*T_marker_mocap
            T_tool_base = T_marker_base*robot_weld.T_tool_toolmarker
            this_mocap_p.append(T_tool_base.p)
            this_mocap_R.append(np.degrees(R2rpy(mocap_R[k])))
            this_orientation_error.append(np.degrees(R2rpy(robot_T.R.T@T_tool_base.R)))

        mocap_position.append(np.mean(this_mocap_p,axis=0))
        robt_position.append(robot_T.p)
        position_error.append(np.mean(this_mocap_p,axis=0)-robot_T.p)
        std_pos_N.append(np.std(this_mocap_p,axis=0))
        std_pos_norm_N.append(np.std(np.linalg.norm(this_mocap_p,2,axis=1)))
        
        orientation_error.append(np.mean(this_orientation_error,axis=0))
        std_ori_N.append(np.std(this_mocap_R,axis=0))
        std_ori_norm_N.append(np.std(np.linalg.norm(this_mocap_R,2,axis=1)))

    print("Testing Set on Pose",testing_pose)
    print("Mean Position:",np.mean(mocap_position,axis=0))
    print("Mean FK Position:",np.mean(robt_position,axis=0))
    print("Mean Position Error Vec:",np.mean(position_error,axis=0))
    print("Mean Abs Position Error Vec:",np.mean(np.fabs(position_error),axis=0))
    print("Mean Position Error:",np.mean(np.linalg.norm(position_error,2,axis=1)))
    print("Std Position:",np.std(mocap_position,axis=0))
    print("Std FK Position:",np.std(robt_position,axis=0))
    print("Std Position Error:",np.std(position_error,axis=0))
    print("Std Position Error Norm:",np.std(np.linalg.norm(position_error,2,axis=1)))
    print("Mean N Std Position:",np.mean(std_pos_N,axis=0))
    print("Mean N Std Position Norm:",np.mean(std_pos_norm_N,axis=0))
    print("===")
    print("Mean Orientation Error Vec:",np.mean(orientation_error,axis=0))
    print("Mean Orientation Error:",np.mean(np.linalg.norm(orientation_error,2,axis=1)))
    print("Std Orientation Error:",np.std(orientation_error,axis=0))
    print("Std Orientation Error Norm:",np.std(np.linalg.norm(orientation_error,2,axis=1)))
    print("Mean N Std Orientation:",np.mean(std_ori_N,axis=0))
    print("Mean N Std Orientation Norm:",np.mean(std_ori_norm_N,axis=0))
    print("===========================")


for pose_N in range(total_pose):
    position_error=[]
    orientation_error=[]
    mocap_position=[]
    mocap_orientation=[]
    robt_position = []
    std_pos_N = []
    std_pos_norm_N = []
    std_ori_N = []
    std_ori_norm_N = []
    for N in range(total_N):

        robot_k = robot_stop_k[N*total_pose+pose_N]
        mocap_k = mocap_stop_k[N*total_pose+pose_N]
        robot_T = robot_weld.fwd(robot_q[robot_k])
        
        this_mocap_p = mocap_p[mocap_k:mocap_k+dK_mocap]
        # # convert to robot base frame
        # this_mocap_p = np.matmul(T_mocap_base.R,this_mocap_p.T).T + T_mocap_base.p
        # convert to basemarker than base
        T_mocap_basemarker = Transform(base_rigid_R[mocap_k],base_rigid_p[mocap_k]).inv()
        this_mocap_p = []
        this_mocap_R = []
        this_orientation_error = []
        for k in range(mocap_k,mocap_k+dK_mocap):
            T_marker_mocap = Transform(mocap_R[k],mocap_p[k])
            T_marker_base = T_basemarker_base*T_mocap_basemarker*T_marker_mocap
            T_tool_base = T_marker_base*robot_weld.T_tool_toolmarker
            this_mocap_p.append(T_tool_base.p)
            this_mocap_R.append(np.degrees(R2rpy(mocap_R[k])))
            this_orientation_error.append(np.degrees(R2rpy(robot_T.R.T@T_tool_base.R)))
        
        # print(this_mocap_R[-1])
        # print(robot_T.R)
        # print(np.mean(this_mocap_p,axis=0))
        # print(robot_T.p)

        # this_mocap_p = []
        # for k in range(mocap_k,mocap_k+dK_mocap):
        #     T_mocap_basemarker = Transform(base_rigid_R[k],base_rigid_p[k]).inv()
        #     p_sample = np.matmul(T_mocap_basemarker.R,mocap_p[k]) + T_mocap_basemarker.p
        #     p_sample = np.matmul(T_basemarker_base.R,p_sample) + T_basemarker_base.p
        #     this_mocap_p.append(p_sample)

        mocap_position.append(np.mean(this_mocap_p,axis=0))
        robt_position.append(robot_T.p)
        position_error.append(np.mean(this_mocap_p,axis=0)-robot_T.p)
        std_pos_N.append(np.std(this_mocap_p,axis=0))
        std_pos_norm_N.append(np.std(np.linalg.norm(this_mocap_p,2,axis=1)))
        
        orientation_error.append(np.mean(this_orientation_error,axis=0))
        std_ori_N.append(np.std(this_mocap_R,axis=0))
        std_ori_norm_N.append(np.std(np.linalg.norm(this_mocap_R,2,axis=1)))

        # print("This N Std Position:",std_pos_N[-1])
        # print("This N Std Position Error:",std_pos_norm_N[-1])

    print("Pose",pose_N)
    print("Mean Position:",np.mean(mocap_position,axis=0))
    print("Mean FK Position:",np.mean(robt_position,axis=0))
    print("Mean Position Error Vec:",np.mean(position_error,axis=0))
    print("Mean Position Error:",np.mean(np.linalg.norm(position_error,2,axis=1)))
    print("Std Position:",np.std(mocap_position,axis=0))
    print("Std FK Position:",np.std(robt_position,axis=0))
    print("Std Position Error:",np.std(position_error,axis=0))
    print("Std Position Error Norm:",np.std(np.linalg.norm(position_error,2,axis=1)))
    print("Mean N Std Position:",np.mean(std_pos_N,axis=0))
    print("Mean N Std Position Norm:",np.mean(std_pos_norm_N,axis=0))
    print("===")
    print("Mean Orientation Error Vec:",np.mean(orientation_error,axis=0))
    print("Mean Orientation Error:",np.mean(np.linalg.norm(orientation_error,2,axis=1)))
    print("Std Orientation Error:",np.std(orientation_error,axis=0))
    print("Std Orientation Error Norm:",np.std(np.linalg.norm(orientation_error,2,axis=1)))
    print("Mean N Std Orientation:",np.mean(std_ori_N,axis=0))
    print("Mean N Std Orientation Norm:",np.mean(std_ori_norm_N,axis=0))
    print("===========================")

    # plt.plot(np.fabs(position_error)[:,0],'-o',label='error x')
    # plt.plot(np.fabs(position_error)[:,1],'-o',label='error y')
    # plt.plot(np.fabs(position_error)[:,2],'-o',label='error z')
    # plt.legend()
    # plt.show()

    # plt.plot(np.fabs(orientation_error)[:,0],'-o',label='error x')
    # plt.plot(np.fabs(orientation_error)[:,1],'-o',label='error y')
    # plt.plot(np.fabs(orientation_error)[:,2],'-o',label='error z')
    # plt.legend()
    # plt.show()