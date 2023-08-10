import numpy as np
from copy import deepcopy
from general_robotics_toolbox import *
import pickle
import time
import sys
# sys.path.append('../toolbox/')
# from robot_def import *
from matplotlib import pyplot as plt

from numpy.random import default_rng

dataset_date='0801'
robot_type='R1'

data_dir='test'+dataset_date+'_'+robot_type+'/train_data_'

base_T = np.loadtxt(data_dir+'base_T_raw.csv',delimiter=',')
tool_T = np.loadtxt(data_dir+'tool_T_raw.csv',delimiter=',')

for tT in tool_T:
    print(np.degrees(R2rpy(q2R(tT[3:]))))
    input("============================")

