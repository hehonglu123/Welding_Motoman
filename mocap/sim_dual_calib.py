import numpy as np
from copy import deepcopy
from general_robotics_toolbox import *
import pickle
import time
import sys
sys.path.append('../toolbox/')
from robot_def import *
from matplotlib import pyplot as plt
from scipy.optimize import fminbound
from qpsolvers import solve_qp
from calib_analytic_grad import *

from numpy.random import default_rng
rng = default_rng()

Rx=np.array([1,0,0])
Ry=np.array([0,1,0])
Rz=np.array([0,0,1])

r1dataset_date='0801'
r2dataset_date='0804'
config_dir='../config/'

###### define robots ######
robots = []

robot_marker_dir=config_dir+'MA2010_marker_config/'
tool_marker_dir=config_dir+'weldgun_marker_config/'
robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',\
                    tool_file_path=config_dir+'torch.csv',d=15,\
                    pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
                    base_marker_config_file=robot_marker_dir+'MA2010_'+r1dataset_date+'_marker_config.yaml',\
                    tool_marker_config_file=tool_marker_dir+'weldgun_'+r1dataset_date+'_marker_config.yaml')
robots.append(robot)
robot_marker_dir=config_dir+'MA1440_marker_config/'
tool_marker_dir=config_dir+'mti_marker_config/'
robot=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',\
                    tool_file_path=config_dir+'mti.csv',\
                    pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
                    base_marker_config_file=robot_marker_dir+'MA1440_'+r2dataset_date+'_marker_config.yaml',\
                    tool_marker_config_file=tool_marker_dir+'mti_'+r2dataset_date+'_marker_config.yaml')
robots.append(robot)
###########################

##### nominal parameters #####
param_noms = []
for robot in robots:
    jN = len(robot.robot.H[0])
    robot.P_nominal=deepcopy(robot.robot.P)
    robot.H_nominal=deepcopy(robot.robot.H)
    robot.P_nominal=robot.P_nominal.T
    robot.H_nominal=robot.H_nominal.T
    robot = get_H_param_axis(robot)
    
    ### nomnial parameters
    param_nom = np.zeros_like(robot.P_nominal)
    param_nom = np.reshape(param_nom, (param_nom.size, ))
    print(param_nom.shape)
##############################

##### simulated actual parameters #####
