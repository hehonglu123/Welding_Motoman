from copy import deepcopy
import sys
sys.path.append('../toolbox/')
sys.path.append('../redundancy_resolution/')
from utils import *
from robot_def import *
from dx200_motion_program_exec_client import *
from MotionSendMotoman import *

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
from threading import Thread
import numpy as np
import time
import pickle
from MocapPoseListener import *

config_dir='../config/'
# robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',d=15,\
# pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg.csv',\
# base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',d=15,\
pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
base_marker_config_file=config_dir+'MA2010_marker_config_rmsecalib.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

test_qs = np.array([[0.,0.,0.,0.,0.,0.],[0,69,57,0,0,0],[0,-68,-68,0,0,0],[-36.6018,12.4119,-12.1251,-43.3579,-45.4297,68.1203],
                [21.0753,-1.8803,-27.3509,13.1122,-25.1173,-25.2466]])
# test_qs = np.array([[0,69,57,0,0,0],[0,-68,-68,0,0,0]])
# print(robot_weld.fwd(np.radians([-0.5,-68,-68,0,0,0])))
# exit()

# mocap pose listener
mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
mocap_url = mocap_url
mocap_cli = RRN.ConnectService(mocap_url)
all_ids=[]
all_ids.extend(robot_weld.tool_markers_id)
all_ids.extend(robot_weld.base_markers_id)
all_ids.append(robot_weld.base_rigid_id)
all_ids.append(robot_weld.tool_rigid_id)
mpl_obj = MocapFrameListener(mocap_cli,all_ids,'world')

data_dir = 'kinematic_raw_data/'

repeats_N = 20
rob_speed = 15

robot_client = MotionProgramExecClient()
mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=robot_weld.pulse2deg)
for N in range(repeats_N):
    for test_q in test_qs:
        # move robot
        mp.MoveJ(test_q,rob_speed,0)
        mp.setWaitTime(3)

# Run
mpl_obj.run_pose_listener()
robot_stamps,curve_exe, job_line,job_step = robot_client.execute_motion_program(mp)
mpl_obj.stop_pose_listener()
curve_p,curve_R,timestamps = mpl_obj.get_frames_traj()

with open(data_dir+'mocap_p_cont.pickle', 'wb') as handle:
    pickle.dump(curve_p, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(data_dir+'mocap_R_cont.pickle', 'wb') as handle:
    pickle.dump(curve_R, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(data_dir+'robot_q_cont.pickle', 'wb') as handle:
    pickle.dump(curve_exe, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(data_dir+'mocap_p_timestamps_cont.pickle', 'wb') as handle:
    pickle.dump(timestamps, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(data_dir+'robot_q_timestamps_cont.pickle', 'wb') as handle:
    pickle.dump(robot_stamps, handle, protocol=pickle.HIGHEST_PROTOCOL)