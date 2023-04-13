import numpy as np
from general_robotics_toolbox import *

# robot_T = np.loadtxt('compare_results_ctrl_basefix.csv',delimiter=',')
# mocap_T = np.loadtxt('compare_results_mocap_basefix.csv',delimiter=',')
robot_T = np.loadtxt('compare_results_ctrl_basefix_toolT.csv',delimiter=',')
mocap_T = np.loadtxt('compare_results_mocap_basefix_toolT.csv',delimiter=',')
total_pose= 2

for j in range(total_pose):
    position_error=[]
    mocap_position=[]
    orientation_error=[]
    mocap_orientation=[]
    for i in range(0,len(robot_T),total_pose):
        print(i)
        rob_R = q2R(robot_T[i+j,3:])
        moc_R = q2R(mocap_T[i+j,3:])
        mocap_position.append(mocap_T[i+j,:3])
        position_error.append(mocap_T[i+j,:3]-robot_T[i+j,:3])
        orientation_error.append(np.degrees(R2rpy(rob_R.T@moc_R)))
        mocap_orientation.append(np.degrees(R2rpy(moc_R)))

    print("Pose",j)
    print("Mean Position:",np.mean(mocap_position,axis=0))
    print("Mean Position Error Vec:",np.mean(position_error,axis=0))
    print("Mean Position Error:",np.mean(np.linalg.norm(position_error,2,axis=1)))
    print("Std Position:",np.std(mocap_position,axis=0))
    print("Std Position Error:",np.std(np.linalg.norm(position_error,2,axis=1)))

    print("Mean Orientation:",np.mean(mocap_orientation,axis=0))
    print("Mean Orientation Error Vec:",np.mean(orientation_error,axis=0))
    print("Mean Orientation Error:",np.mean(np.linalg.norm(orientation_error,2,axis=1)))
    print("Std Orientation:",np.std(orientation_error,axis=0))
    print("Std Orientation Error:",np.std(np.linalg.norm(orientation_error,2,axis=1)))

    # print(np.array(orientation_error))