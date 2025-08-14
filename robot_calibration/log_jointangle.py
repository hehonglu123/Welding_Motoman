from copy import deepcopy
import sys
from robotics_utils import *
from motoman_def  import *

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
from threading import Thread
import numpy as np
import time
import yaml
from dx200_motion_program_exec_client import *
from WeldSend import *

robot_client=MotionProgramExecClient()
ws=WeldSend(robot_client)

def get_joint_angle():
    for attempt_i in range(10):
        res, fb_data = ws.client.fb.try_receive_state_sync(ws.client.controller_info, 0.001)
        if attempt_i < 5:
            continue
        if res:
            joint_angle=np.hstack((fb_data.group_state[0].feedback_position,fb_data.group_state[1].feedback_position,fb_data.group_state[2].feedback_position))
            return joint_angle
    return None

filename = 'joint_angles.csv'

try:
    joint_angles = np.loadtxt(filename, delimiter=',')
except FileNotFoundError:
    joint_angles = []
    pass

if len(joint_angles) == 0:
    joint_angles = [np.degrees(get_joint_angle())]
else:
    joint_angles = np.vstack((joint_angles, np.degrees(get_joint_angle())))

np.savetxt(filename, joint_angles, delimiter=',')