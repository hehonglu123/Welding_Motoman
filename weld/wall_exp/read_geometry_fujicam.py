from copy import deepcopy
from pathlib import Path
import pickle
import sys
sys.path.append('../')
sys.path.append('../../toolbox/')
sys.path.append('../../scan/scan_tools/')
sys.path.append('../../scan/scan_plan/')
sys.path.append('../../scan/scan_process/')
from robot_def import *
from scan_utils import *
from scan_continuous import *
# from scanPathGen import *
from scanProcess import *
from weldCorrectionStrategy import *
from weldRRSensor import *
from WeldSend import *
from dx200_motion_program_exec_client import *

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import yaml
import open3d as o3d

zero_config=np.zeros(6)
# 0. robots. Note use "(robot)_pose_mocapcalib.csv"
config_dir='../../config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'torch.csv',\
    pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA2010_marker_config/MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config/weldgun_marker_config.yaml')
robot_scan=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'fujicam.csv',\
    pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
robot_flir=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
	pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'D500B_marker_config/D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config/positioner_tcp_marker_config.yaml')

data_dir='../../data/wall_weld_test/torch_ori0_2024_09_24_15_46_20/'

# 1. read meta file
meta_file=data_dir+'meta.yaml'
with open(meta_file) as file:
    meta_info=yaml.load(file,Loader=yaml.FullLoader)

for layer_i in range(meta_info['total_weld_layer']):

    # get layer data dir
    layer_dir = data_dir + 'weldlayer_' + str(layer_i) + '/'
    # get stamps, robot scan js and scan
    stamps_scan = np.loadtxt(layer_dir + 'timestamps_scan.csv', delimiter=',')
    robot_scan_js = np.loadtxt(layer_dir + 'robot_scan_js.csv', delimiter=',')
    positioner_js = np.loadtxt(layer_dir + 'positioner_js.csv', delimiter=',')
    with open(layer_dir + 'scan.pkl', 'rb') as f:
        scan = pickle.load(f)

    print(scan[0])
    print(len(scan))
    print(len(robot_scan_js))
    input('')