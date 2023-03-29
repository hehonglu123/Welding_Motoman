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

print(rot([0,1,0],np.radians(22)))
exit()

config_dir='../config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',\
	base_transformation_file=config_dir+'MA2010_mocap_pose.csv',d=20,pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg.csv')
T_base_world = np.linalg.inv(robot_weld.base_H)
T_base_world = Transform(T_base_world[:3,:3],T_base_world[:3,-1])

marker_ids = ['rigid2']
mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'

test_q = [np.array([0.,0.,0.,0.,0.,0.])]

mocap_cli = RRN.ConnectService(mocap_url)
robot_client=MotionProgramExecClient(ROBOT_CHOICE='RB1',pulse2deg=robot_weld.pulse2deg)

robot_client.MoveJ(test_q[0],3,0)
robot_client.ProgEnd()
robot_stamps,curve_exe = robot_client.execute_motion_program("AAA.JBI")
encoder_T = robot_weld.fwd(curve_exe[-1,:6])

sensor_data_srv = mocap_cli.fiducials_sensor_data.Connect(-1)
data = sensor_data_srv.ReceivePacketWait(timeout=10)
for i in range(len(data.fiducials.recognized_fiducials)):
    this_marker_id = data.fiducials.recognized_fiducials[i].fiducial_marker
    if this_marker_id in marker_ids:
        this_position = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['position']))
        this_orientation = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['orientation']))
        mocap_T = Transform(q2R(this_orientation),this_position)
        mocap_T = T_base_world*mocap_T

print(encoder_T)
print(mocap_T)

