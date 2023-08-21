import numpy as np
from copy import deepcopy
import sys
sys.path.append('../../toolbox/')
from utils import *
from robot_def import * 

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
from threading import Thread
import numpy as np
import time
import yaml
from fanuc_motion_program_exec_client import *
from MocapPoseListener import *
import pickle

robot_type='R1'

if robot_type=='R1':
    config_dir='../../config/'
    robot_name='M10ia'
    tool_name='ge_R1_tool'
elif robot_type=='R2':
    config_dir='../../config/'
    robot_name='LRMATE200id'
    tool_name='ge_R2_tool'

robot_marker_dir=config_dir+robot_name+'_marker_config/'
tool_marker_dir=config_dir+tool_name+'_marker_config/'
robot=robot_obj(robot_name,def_path=config_dir+robot_name+'_robot_default_config.yml',tool_file_path=config_dir+tool_name+'.csv',\
base_marker_config_file=robot_marker_dir+robot_name+'_marker_config.yaml',tool_marker_config_file=tool_marker_dir+tool_name+'_marker_config.yaml')

rob_ip='127.0.0.1'

data_dir='single_data/'
data_name=data_dir+'pointer_'+'1'

mocap_url = 'rr+tcp://localhost:59823?service=phasespace_mocap'
mocap_cli = RRN.ConnectService(mocap_url)
all_ids=[]
all_ids.extend(robot.tool_markers_id)
all_ids.extend(robot.base_markers_id)
all_ids.append(robot.base_rigid_id)
all_ids.append(robot.tool_rigid_id)

mpl_obj = MocapFrameListener(mocap_cli,all_ids,'world',use_quat=True)

client = FANUCClient(rob_ip)

mpl_obj.run_pose_listener()
curve_exe = client.get_joint_angle(read_N=3)
mpl_obj.stop_pose_listener()
curve_p,curve_R,timestamps,curve_cond = mpl_obj.get_frames_traj_cond()

with open(data_name+'_mocap_p.pickle', 'wb') as handle:
    pickle.dump(curve_p, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(data_name+'_mocap_R.pickle', 'wb') as handle:
    pickle.dump(curve_R, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(data_name+'_mocap_timestamps.pickle', 'wb') as handle:
    pickle.dump(timestamps, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(data_name+'_mocap_cond.pickle', 'wb') as handle:
    pickle.dump(curve_cond, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(data_name+'_robot_q.pickle', 'wb') as handle:
    pickle.dump(curve_exe, handle, protocol=pickle.HIGHEST_PROTOCOL)
