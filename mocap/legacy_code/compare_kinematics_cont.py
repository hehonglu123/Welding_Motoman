import numpy as np
from copy import deepcopy
from general_robotics_toolbox import *
import pickle
import sys
sys.path.append('../toolbox/')
from robot_def import *
from matplotlib import pyplot as plt

from numpy.random import default_rng
rng = default_rng()

Rx=np.array([1,0,0])
Ry=np.array([0,1,0])
Rz=np.array([0,0,1])

config_dir='../config/'

robot_weld_collect=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path='',d=0,\
pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
base_marker_config_file=config_dir+'MA2010_0613_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path='',d=0,\
pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
base_marker_config_file=config_dir+'MA2010_0613_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')
# robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',\
#                      tool_file_path=config_dir+'torch.csv',d=15,\
# pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
# base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

T_base_basemarker = robot_weld.T_base_basemarker
T_basemarker_base = T_base_basemarker.inv()
robot_weld.T_tool_toolmarker=Transform(np.eye(3),[0,0,0])
robot_weld.robot.T_flange = robot_weld.T_tool_flange
#### using tool
robot_weld.robot.R_tool = robot_weld.T_tool_toolmarker.R
robot_weld.robot.p_tool = robot_weld.T_tool_toolmarker.p

data_dir='kinematic_raw_data/test0620_aftercalib/'

try:
    robot_q = np.loadtxt(data_dir+'robot_q_align.csv',delimiter=',')
    mocap_T = np.loadtxt(data_dir+'mocap_T_align_origin.csv',delimiter=',')

    mocap_T_actual=[]
    for mT in mocap_T:
        T_toolrigid_base_collect=Transform(q2R(mT[3:]),mT[:3])
        T_toolrigid_basemarker = robot_weld_collect.T_base_basemarker*T_toolrigid_base_collect
        T_toolrigid_base = T_basemarker_base*T_toolrigid_basemarker
        mocap_T_actual.append(np.append(T_toolrigid_base.p,R2q(T_toolrigid_base.R)))

    mocap_T_actual = np.array(mocap_T_actual)
    np.savetxt(data_dir+'mocap_T_align.csv',mocap_T_actual,delimiter=',')
    exit()

except:

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
    with open(data_dir+'mocap_quat_cont.pickle', 'rb') as handle:
        mocap_R = pickle.load(handle)
        # static_mocap_marker = mocap_p[robot_weld.base_markers_id[0]]
        base_rigid_R = mocap_R[robot_weld.base_rigid_id]
        mocap_R = np.array(mocap_R[marker_id])
    with open(data_dir+'mocap_p_timestamps_cont.pickle', 'rb') as handle:
        mocap_stamps = pickle.load(handle)
        base_rigid_stamps = np.array(mocap_stamps[marker_id])
        mocap_stamps = np.array(mocap_stamps[marker_id])
    mocap_pdot = np.divide(np.gradient(mocap_p,axis=0),np.tile(np.gradient(mocap_stamps),(3,1)).T)
    mocap_pdot_norm = np.linalg.norm(mocap_pdot,axis=1)

    print(len(mocap_p))
    print(len(mocap_stamps))
    print(len(base_rigid_p))

    mocap_start_k = 3000
    mocap_R = mocap_R[mocap_start_k:]
    mocap_p = mocap_p[mocap_start_k:]
    mocap_stamps = mocap_stamps[mocap_start_k:]
    mocap_pdot = mocap_pdot[mocap_start_k:]
    mocap_pdot_norm = mocap_pdot_norm[mocap_start_k:]
    base_rigid_p=base_rigid_p[mocap_start_k:]
    base_rigid_R=base_rigid_R[mocap_start_k:]
    base_rigid_stamps=base_rigid_stamps[mocap_start_k:]

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

    mocap_vdev_thres = 5
    mocap_v_thres = 8
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
        if mocap_v_dev[-1]>=mocap_vdev_thres:
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

    assert len(robot_stop_k)==len(mocap_stop_k), f"Mocap Stop and Robot Stop should be the same."

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

