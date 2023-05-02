import numpy as np
from copy import deepcopy
from general_robotics_toolbox import *
import pickle
import sys
sys.path.append('../../toolbox/')
from robot_def import *
from matplotlib import pyplot as plt

config_dir='../../config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path='',d=0,\
pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

raw_data_dir = 'test0502_noanchor/'
for j in range(6):
    with open(raw_data_dir+'mocap_p_cont.pickle', 'rb') as handle:
        curve_p = pickle.load(handle)
    with open(raw_data_dir+'mocap_R_cont.pickle', 'rb') as handle:
        curve_R = pickle.load(handle)
    with open(raw_data_dir+'mocap_p_timestamps_cont.pickle', 'rb') as handle:
        mocap_stamps = pickle.load(handle)
    print(curve_p[robot_weld.tool_rigid_id][0])
    print(curve_R[robot_weld.tool_rigid_id][0])
    print(mocap_stamps[robot_weld.tool_rigid_id][0])
exit()

data_dir='./'
with open(data_dir+'robot_q_cont.pickle', 'rb') as handle:
    robot_q = pickle.load(handle)
    robot_q = robot_q[:,:6]
with open(data_dir+'robot_q_timestamps_cont.pickle', 'rb') as handle:
    robot_q_stamps = pickle.load(handle)
robot_qdot = np.divide(np.gradient(robot_q,axis=0),np.tile(np.gradient(robot_q_stamps),(6,1)).T)
robot_qdot_norm = np.linalg.norm(robot_qdot,axis=1)

# plt.plot(robot_qdot_norm)
# plt.plot(np.diff(robot_q_stamps))
# plt.show()

marker_id = 'marker4_rigid3'
data_dir='./'
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

# print(len(mocap_p))
# print(len(base_rigid_p))

mocap_start_k = 400
mocap_p = mocap_p[mocap_start_k:]
mocap_p_stamps = mocap_p_stamps[mocap_start_k:]
mocap_pdot = mocap_pdot[mocap_start_k:]
mocap_pdot_norm = mocap_pdot_norm[mocap_start_k:]
base_rigid_p=base_rigid_p[mocap_start_k:]
base_rigid_R=base_rigid_R[mocap_start_k:]

# plt.plot(mocap_pdot_norm)
# plt.plot(np.diff(mocap_p_stamps))
# plt.show()

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
# plt.plot(robot_v_dev)
# plt.scatter(robot_stop_k,robot_v_dev[robot_stop_k])
# plt.plot(robot_qdot_norm,'blue')
# for k in robot_stop_k:
#     plt.plot(np.arange(k,k+dK_robot),robot_qdot_norm[k:k+dK_robot])
# plt.show()


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
# plt.plot(mocap_v_dev)
# plt.plot(mocap_pdot_norm,'blue')
# for k in mocap_stop_k:
#     plt.plot(np.arange(k,k+dK_mocap),mocap_pdot_norm[k:k+dK_mocap])
# plt.show()

# static_mocap_marker = np.array(static_mocap_marker)
# print(np.std(static_mocap_marker,axis=0))
# # print(static_mocap_marker)
# for i in range(3):
#     plt.plot(static_mocap_marker[:,i]-np.mean(static_mocap_marker[:,i]))
# plt.show()
# exit()

total_pose = 5
total_N = int(len(mocap_stop_k)/total_pose)

# change robot PH to calib PH
robot_weld.robot.P = robot_weld.calib_P
robot_weld.robot.H = robot_weld.calib_H
T_mocap_base = robot_weld.T_base_mocap.inv()
T_base_basemarker = Transform(base_rigid_R[0],base_rigid_p[0]).inv()*robot_weld.T_base_mocap
T_basemarker_base = T_base_basemarker.inv()

marker_id = 'marker4_rigid3'
for pose_N in range(total_pose):
    position_error=[]
    mocap_position=[]
    robt_position = []
    std_pos_N = []
    std_pos_norm_N = []
    for N in range(total_N):

        robot_k = robot_stop_k[N*total_pose+pose_N]
        mocap_k = mocap_stop_k[N*total_pose+pose_N]

        robot_T = robot_weld.fwd(robot_q[robot_k])
        this_mocap_p = mocap_p[mocap_k:mocap_k+dK_mocap]
        # # convert to robot base frame
        # this_mocap_p = np.matmul(T_mocap_base.R,this_mocap_p.T).T + T_mocap_base.p

        # convert to basemarker than base
        T_mocap_basemarker = Transform(base_rigid_R[mocap_k],base_rigid_p[mocap_k]).inv()
        this_mocap_p = np.matmul(T_mocap_basemarker.R,this_mocap_p.T).T + T_mocap_basemarker.p
        this_mocap_p = np.matmul(T_basemarker_base.R,this_mocap_p.T).T + T_basemarker_base.p

        mocap_position.append(np.mean(this_mocap_p,axis=0))
        robt_position.append(robot_T.p)
        position_error.append(np.mean(this_mocap_p,axis=0)-robot_T.p)
        std_pos_N.append(np.std(this_mocap_p,axis=0))
        std_pos_norm_N.append(np.std(np.linalg.norm(this_mocap_p,2,axis=1)))

        # print("This N Std Position:",std_pos_N[-1])
        # print("This N Std Position Error:",std_pos_norm_N[-1])

    plt.plot(np.fabs(position_error)[:,0],'-o',label='error x')
    plt.plot(np.fabs(position_error)[:,1],'-o',label='error y')
    plt.plot(np.fabs(position_error)[:,2],'-o',label='error z')
    plt.legend()
    plt.show()

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
    print("Mean N Std Position Error:",np.mean(std_pos_norm_N,axis=0))
    print("===========================")