import numpy as np
from copy import deepcopy
from general_robotics_toolbox import *
from pathlib import Path
import glob

import sys
sys.path.append('../toolbox/')
from robot_def import *
from matplotlib import pyplot as plt
from dx200_motion_program_exec_client import *

data_dir='tool_data/'+'R2_mti_0810/'

ph_dataset_date='0801'

config_dir='../config/'

robot_type = 'R2'

data_type='zaxis'

if robot_type == 'R1':
    robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',\
                        tool_file_path=config_dir+'torch.csv',d=15,\
                        #  tool_file_path='',d=0,\
                        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
                        base_marker_config_file=config_dir+'MA2010_'+ph_dataset_date+'_marker_config.yaml',\
                        tool_marker_config_file=config_dir+'weldgun_'+ph_dataset_date+'_marker_config.yaml')
elif robot_type == 'R2':
    robot=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',\
                        tool_file_path=config_dir+'mti.csv',\
                        pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
                        base_marker_config_file=config_dir+'MA1440_'+ph_dataset_date+'_marker_config.yaml',\
                        tool_marker_config_file=config_dir+'mti_'+ph_dataset_date+'_marker_config.yaml')

robot_client=MotionProgramExecClient()

robot_client.StartStreaming()
res, data = robot_client.receive_from_robot(0.01)
if robot_type=='R1':
    joint_angle=np.radians(np.divide(np.array(data[20:26]),robot.pulse2deg))
elif robot_type=='R2':
    joint_angle=np.radians(np.divide(np.array(data[26:32]),robot.pulse2deg))
robot_client.StopStreaming()

Path(data_dir).mkdir(exist_ok=True)
num_js=len(glob.glob(data_dir+'pose_js_'+data_type+'_*.csv'))
np.savetxt(data_dir+'pose_js_'+data_type+'_'+str(num_js)+'.csv',joint_angle,delimiter=',')

print("Total # of Poses:",num_js+1)