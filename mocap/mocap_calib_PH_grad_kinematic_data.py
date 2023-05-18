import numpy as np
from copy import deepcopy
from general_robotics_toolbox import *
import pickle
import sys
sys.path.append('../toolbox/')
from robot_def import *
from matplotlib import pyplot as plt

Rx=np.array([1,0,0])
Ry=np.array([0,1,0])
Rz=np.array([0,0,1])

config_dir='../config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path='',d=0,\
pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
base_marker_config_file=config_dir+'MA2010_0504_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

# data_dir='kinematic_raw_data/test0502_noanchor/'
data_dir='kinematic_raw_data/test0504/'

with open(data_dir+'robot_q_cont.pickle', 'rb') as handle:
    robot_q = pickle.load(handle)
    robot_q = robot_q[:,:6]
with open(data_dir+'robot_q_timestamps_cont.pickle', 'rb') as handle:
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
with open(data_dir+'mocap_p_timestamps_cont.pickle', 'rb') as handle:
    mocap_p_stamps = pickle.load(handle)
    mocap_p_stamps = np.array(mocap_p_stamps[marker_id])
mocap_pdot = np.divide(np.gradient(mocap_p,axis=0),np.tile(np.gradient(mocap_p_stamps),(3,1)).T)
mocap_pdot_norm = np.linalg.norm(mocap_pdot,axis=1)

print(len(mocap_p))
print(len(mocap_p_stamps))
print(len(base_rigid_p))

mocap_start_k = 400
mocap_R = mocap_R[mocap_start_k:]
mocap_p = mocap_p[mocap_start_k:]
mocap_p_stamps = mocap_p_stamps[mocap_start_k:]
mocap_pdot = mocap_pdot[mocap_start_k:]
mocap_pdot_norm = mocap_pdot_norm[mocap_start_k:]
base_rigid_p=base_rigid_p[mocap_start_k:]
base_rigid_R=base_rigid_R[mocap_start_k:]

# the robot stop 3 sec, 
# find "2 sec window" with smallest norm of velocity deviation
robot_dev_thres = 0.01
dt_ave_robot = np.mean(np.gradient(robot_q_stamps))
dK_robot = int(2/dt_ave_robot)
robot_v_dev = []
robot_stop_k = []
v_dev_flag = False
v_dev_local = []
for i in range(0,len(robot_q)-dK_robot):
    robot_v_dev.append(np.std(robot_qdot_norm[i:i+dK_robot]))
    if robot_v_dev[-1]<robot_dev_thres:
        v_dev_flag=True
        v_dev_local.append(robot_v_dev[-1])
    else:
        if v_dev_flag:
            v_dev_flag=False
            local_argmin = np.argmin(v_dev_local)
            robot_stop_k.append(i-len(v_dev_local)+local_argmin)
            v_dev_local=[]
robot_v_dev=np.array(robot_v_dev)
robot_stop_k.append(np.argmin(robot_v_dev[robot_stop_k[-1]+dK_robot:])+robot_stop_k[-1]+dK_robot)

mocap_dev_thres = 5
dt_ave_mocap = np.mean(np.gradient(mocap_p_stamps))
dK_mocap = int(2/dt_ave_mocap)
mocap_v_dev = []
mocap_stop_k = []
v_dev_flag = False
v_dev_local = []
for i in range(0,len(mocap_pdot)-dK_mocap):
    mocap_v_dev.append(np.std(mocap_pdot_norm[i:i+dK_mocap]))
    if mocap_v_dev[-1]<mocap_dev_thres:
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

total_pose = 5
total_N = int(len(mocap_stop_k)/total_pose)

# random select test and train set
setN = np.arange(total_N)
np.random.shuffle(setN)
train_N = setN[:int(total_N/2)]
test_N = setN[int(total_N/2):]

T_base_basemarker = robot_weld.T_base_basemarker
T_basemarker_base = T_base_basemarker.inv()
robot_weld.T_tool_toolmarker=Transform(np.eye(3),[0,0,0])

#### Gradient
print("Training Set:",train_N)
plot_grad=True
all_testing_pose=[1]
# all_testing_pose=[0,1,2,3,4]
total_iteration = 100
total_grad_sample = 40
dP_range = 0.05
dH_range = np.radians(0.1)
dH_rotate_axis = [[Rx,Ry],[Rz,Rx],[Rz,Rx],[Ry,Rz],[Rz,Rx],[Ry,Rz]]
alpha=0.01
pos_error_progress = []
for iter_N in range(total_iteration):
    print(iter_N)
    G_ave = []
    position_error=[]
    orientation_error=[]
    for N in train_N:
        d_T_all = [] # difference in robot T
        d_pH_all = [] # difference in PH
        for testing_pose in all_testing_pose:
            # initial robot_T
            robot_k = robot_stop_k[N*total_pose+testing_pose]
            robot_weld.robot.P = deepcopy(robot_weld.calib_P)
            robot_weld.robot.H = deepcopy(robot_weld.calib_H)
            robot_weld.robot.T_flange = robot_weld.T_tool_flange
            robot_init_T = robot_weld.fwd(robot_q[robot_k])
            for sample_N in range(total_grad_sample):
                
                # change robot PH to perturb calib PH
                dP=np.random.uniform(low=-dP_range,high=dP_range,size=robot_weld.calib_P.T.shape)
                robot_weld.robot.P = robot_weld.calib_P+dP.T
                dH = np.zeros((6,2))
                for i in range(len(dH)):
                    new_H = deepcopy(robot_weld.calib_H[:,i])
                    for j in range(2):
                        d_angle = np.random.uniform(low=-dH_range,high=dH_range)
                        new_H = np.matmul(rot(dH_rotate_axis[i][j],d_angle),new_H)
                        dH[i,j]=np.degrees(d_angle)
                    robot_weld.robot.H[:,i] = new_H
                
                robot_pert_T = robot_weld.fwd(robot_q[robot_k])
                dR = robot_init_T.R.T@robot_pert_T.R
                k,theta = R2rot(dR)
                dPos = robot_pert_T.p-robot_init_T.p
                if theta == 0:
                    ktheta=np.zeros(3)
                else:
                    ktheta=k*np.degrees(theta)
                d_T_all.append(np.append(dPos,ktheta))
                d_pH_all.append(np.append(dP.flatten(),dH.flatten()))
        d_T_all = np.array(d_T_all).T
        d_pH_all = np.array(d_pH_all).T
        G = np.matmul(d_T_all,np.linalg.pinv(d_pH_all))
        G_ave.append(G)

        # get error
        mocap_k = mocap_stop_k[N*total_pose+testing_pose]
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
            k,theta = R2rot(robot_init_T.R.T@T_tool_base.R)
            k=np.array(k)
            this_orientation_error.append(k*np.degrees(theta))
        
        position_error.append(np.mean(this_mocap_p,axis=0)-robot_init_T.p)
        orientation_error.append(np.mean(this_orientation_error,axis=0))

    G_ave=np.mean(G_ave,axis=0)
    if (iter_N==0 or iter_N==total_iteration-1) and plot_grad:
        print(G_ave)
        plt.matshow(G_ave)
        plt.colorbar()
        plt.show()
    position_error = np.mean(position_error,axis=0)
    orientation_error = np.mean(orientation_error,axis=0)
    
    # update PH
    error_pos_ori = np.append(position_error,orientation_error)
    d_pH_update = alpha*np.matmul(np.linalg.pinv(G_ave),error_pos_ori)
    robot_weld.calib_P = robot_weld.calib_P+np.reshape(d_pH_update[:robot_weld.calib_P.size],robot_weld.calib_P.T.shape).T
    dH = np.reshape(d_pH_update[robot_weld.calib_P.size:],(6,2))
    for i in range(len(dH)):
        new_H = deepcopy(robot_weld.calib_H[:,i])
        for j in range(2):
            d_angle = dH[i,j]
            new_H = np.matmul(rot(dH_rotate_axis[i][j],np.radians(d_angle)),new_H)
        robot_weld.calib_H[:,i] = new_H

    pos_error_progress.append(position_error)

plt.plot(pos_error_progress)
plt.show()

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