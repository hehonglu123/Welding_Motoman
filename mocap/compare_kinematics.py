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
from MocapPoseListener import *

config_dir='../config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',\
pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg.csv',\
base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

test_qs = np.array([[0.,0.,0.,0.,0.,0.],[0,69,57,0,0,0],[0,-68,-68,0,0,0],[-36.6018,12.4119,-12.1251,-43.3579,-45.4297,68.1203],
                [21.0753,-1.8803,-27.3509,13.1122,-25.1173,-25.2466]])

# mocap pose listener
mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
mocap_url = mocap_url
mocap_cli = RRN.ConnectService(mocap_url)
mpl_obj = MocapPoseListener(mocap_cli,[robot_weld])

all_robot_ctrl_pose = []
all_robot_mocap_pose = []
for test_q in test_qs:
    # move robot
    robot_client = MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB1',pulse2deg=robot_weld.pulse2deg)
    robot_client.MoveJ(test_q,5,0)
    robot_client.ProgEnd()
    robot_stamps,curve_exe = robot_client.execute_motion_program("AAA.JBI")
    encoder_T = robot_weld.fwd(curve_exe[-1,:6])
    all_robot_ctrl_pose.append(np.append(encoder_T.p,R2q(encoder_T.R)))

    mpl_obj.run_pose_listener()
    time.sleep(1)
    mpl_obj.stop_pose_listener()
    curve_p,curve_R,timestamps = mpl_obj.get_robots_traj()
    mocap_T = Transform(curve_R[robot_weld.robot_name][-1],curve_p[robot_weld.robot_name][-1])
    all_robot_mocap_pose.append(np.append(mocap_T.p,R2q(mocap_T.R)))

# move robot
robot_client = MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB1',pulse2deg=robot_weld.pulse2deg)
robot_client.MoveJ(test_qs[2],5,0)
robot_client.ProgEnd()
robot_stamps,curve_exe = robot_client.execute_motion_program("AAA.JBI")
encoder_T = robot_weld.fwd(curve_exe[-1,:6])

mpl_obj.run_pose_listener()
time.sleep(1)
mpl_obj.stop_pose_listener()
curve_p,curve_R,timestamps = mpl_obj.get_robots_traj()
mocap_T = Transform(curve_R[robot_weld.robot_name][-1],curve_p[robot_weld.robot_name][-1])

print(encoder_T)
print(mocap_T)

np.savetxt('data/compare_results_ctrl.csv',all_robot_ctrl_pose,delimiter=',')
np.savetxt('data/compare_results_mocap.csv',all_robot_mocap_pose,delimiter=',')

