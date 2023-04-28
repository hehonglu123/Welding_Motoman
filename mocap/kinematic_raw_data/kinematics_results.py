import numpy as np
from general_robotics_toolbox import *
import pickle
import sys
sys.path.append('../toolbox/')
from robot_def import *

config_dir='../../config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',d=15,\
pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

data_dir='./'
with open(data_dir+'robot_q.pickle', 'rb') as handle:
    robot_q = pickle.load(handle)

total_N = len(robot_q.keys())
total_pose = len(robot_q[0].keys())

marker_id = 'rigid3_marker1'

for pose_N in range(total_pose):
    position_error=[]
    mocap_position=[]
    for N in range(total_N):
        with open(data_dir+'pose'+str(pose_N)+'_N'+str(N)+'.pickle', 'rb') as handle:
            curve_p=pickle.load(handle, protocol=pickle.HIGHEST_PROTOCOL)
        robot_T = robot_weld.fwd(robot_q[N][pose_N])
        mocap_p = curve_p[marker_id][-1]
        mocap_position.append(mocap_p)
        position_error.append(mocap_p-robot_T.p)

    print("Pose",pose_N)
    print("Mean Position:",np.mean(mocap_position,axis=0))
    print("Mean Position Error Vec:",np.mean(position_error,axis=0))
    print("Mean Position Error:",np.mean(np.linalg.norm(position_error,2,axis=1)))
    print("Std Position:",np.std(mocap_position,axis=0))
    print("Std Position Error:",np.std(np.linalg.norm(position_error,2,axis=1)))