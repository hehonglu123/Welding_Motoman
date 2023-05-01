import numpy as np
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

data_dir='./'
with open(data_dir+'robot_q.pickle', 'rb') as handle:
    robot_q = pickle.load(handle)

total_N = len(robot_q.keys())
total_pose = len(robot_q[0].keys())

# change robot PH to calib PH
robot_weld.robot.P = robot_weld.calib_P
robot_weld.robot.H = robot_weld.calib_H
T_mocap_base = robot_weld.T_base_mocap.inv()

marker_id = 'marker4_rigid3'
for pose_N in range(total_pose):
    position_error=[]
    mocap_position=[]
    robt_position = []
    std_pos_N = []
    std_pos_norm_N = []
    for N in range(total_N):
        with open(data_dir+'pose'+str(pose_N)+'_N'+str(N)+'.pickle', 'rb') as handle:
            curve_p=pickle.load(handle)
        robot_T = robot_weld.fwd(robot_q[N][pose_N])

        mocap_p = np.array(curve_p[marker_id]).T
        # convert to robot base frame
        mocap_p = np.matmul(T_mocap_base.R,mocap_p).T + T_mocap_base.p

        mocap_position.append(np.mean(mocap_p,axis=0))
        robt_position.append(robot_T.p)
        position_error.append(np.mean(mocap_p,axis=0)-robot_T.p)
        std_pos_N.append(np.std(mocap_p,axis=0))
        std_pos_norm_N.append(np.std(np.linalg.norm(mocap_p,2,axis=1)))

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