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

filename='PH_rotate_data/train_data_1_mocap_p.pickle'

with open(filename,'rb') as handle:
    data = pickle.load(handle)

print(data.keys())
print(data['rigid1'][:10])