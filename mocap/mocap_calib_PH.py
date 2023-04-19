from copy import deepcopy
import sys
sys.path.append('../toolbox/')
sys.path.append('../redundancy_resolution/')
from utils import *
from robot_def import * 

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
from threading import Thread
import numpy as np
import time
import yaml
from fitting_3dcircle import fitting_3dcircle
from dx200_motion_program_exec_client import *
from MocapPoseListener import *

class CalibRobotPH:
    def __init__(self,mocap_cli,robot) -> None:
        
        self.mocap_cli = mocap_cli

        self.calib_marker_ids = robot.calib_markers_id
        self.base_markers_ids = robot.base_markers_id
        self.base_rigid_id = robot.base_rigid_id

        # mocap listener
        all_ids=[]
        all_ids.extend(self.calib_marker_ids)
        all_ids.extend(self.base_markers_ids)
        all_ids.append(self.base_rigid_id)
        mpl_obj = MocapFrameListener(self.mocap_cli,all_ids,'world')
    
    def run_calib(self,base_marker_config_file,auto=False,rob_IP=None,ROBOT_CHOICE=None,rob_p2d=None,paths=[],rob_speed=3,repeat_N=1):

        pass


def calib_R1():

    config_dir='../config/'
    robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg.csv',base_marker_config_file=config_dir+'MA2010_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
    mocap_cli = RRN.ConnectService(mocap_url)

    j1_rough_axis_direction = np.array([0,1,0])
    j2_rough_axis_direction = np.array([1,0,0])
    calib_obj = CalibRobotPH(mocap_cli,robot_weld.calib_markers_id,robot_weld.base_markers_id,robot_weld.base_rigid_id,j1_rough_axis_direction,j2_rough_axis_direction)


if __name__=='__main__':

    calib_R1()