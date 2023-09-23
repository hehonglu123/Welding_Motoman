from copy import deepcopy
import sys
sys.path.append('../toolbox/')
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

class CalibRobotPH:
    def __init__(self,mocap_cli,robot) -> None:
        
        self.mocap_cli = mocap_cli

        self.robot = robot
        # self.calib_marker_ids = robot.calib_markers_id
        self.calib_marker_ids = robot.tool_markers_id
        print('Calib ID:',self.calib_marker_ids)
        self.base_markers_ids = robot.base_markers_id
        self.base_rigid_id = robot.base_rigid_id

        # mocap listener
        all_ids=[]
        all_ids.extend(self.calib_marker_ids)
        all_ids.extend(self.base_markers_ids)
        all_ids.append(self.base_rigid_id)
        all_ids.append(robot.tool_rigid_id)
        self.mpl_obj = MocapFrameListener(self.mocap_cli,all_ids,'world')

    def run_calib(self,rob_ip=None,start_p=None,paths=[],rob_speed=3,repeat_N=1,
                  raw_data_dir=''):
        
        client = FANUCClient(rob_ip)

        input("Press Enter and the robot will start moving.")
        for j in range(5,-1,-1): # from axis 6 to axis 1

            input("Press Enter and Start Rotating Joint "+str(j+1))
            self.mpl_obj.run_pose_listener()
            input("Press Enter after Done Rotating Joint "+str(j+1))
            self.mpl_obj.stop_pose_listener()
            curve_p,curve_R,timestamps,curve_cond = self.mpl_obj.get_frames_traj_cond()

            with open(raw_data_dir+'_'+str(j+1)+'_mocap_p.pickle', 'wb') as handle:
                pickle.dump(curve_p, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(raw_data_dir+'_'+str(j+1)+'_mocap_R.pickle', 'wb') as handle:
                pickle.dump(curve_R, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(raw_data_dir+'_'+str(j+1)+'_mocap_timestamps.pickle', 'wb') as handle:
                pickle.dump(timestamps, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(raw_data_dir+'_'+str(j+1)+'_mocap_cond.pickle', 'wb') as handle:
                pickle.dump(curve_cond, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # save zero config
        input("Press Enter after moving the robot to zero config")
        print("Reading Zero Config")
        self.mpl_obj.run_pose_listener()
        curve_exe = client.get_joint_angle(read_N=10)
        self.mpl_obj.stop_pose_listener()
        curve_p,curve_R,timestamps,curve_cond = self.mpl_obj.get_frames_traj_cond()
        
        with open(raw_data_dir+'_zero_mocap_p.pickle', 'wb') as handle:
            pickle.dump(curve_p, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(raw_data_dir+'_zero_mocap_R.pickle', 'wb') as handle:
            pickle.dump(curve_R, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(raw_data_dir+'_zero_mocap_timestamps.pickle', 'wb') as handle:
            pickle.dump(timestamps, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(raw_data_dir+'_zero_mocap_cond.pickle', 'wb') as handle:
            pickle.dump(curve_cond, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(raw_data_dir+'_zero_robot_q.pickle', 'wb') as handle:
            pickle.dump(curve_exe, handle, protocol=pickle.HIGHEST_PROTOCOL)

def calib_R2():

    config_dir='../config/'
    robot_name='LRMATE200id'
    tool_name='ge_R2_tool'
    robot_marker_dir=config_dir+robot_name+'_marker_config/'
    tool_marker_dir=config_dir+tool_name+'_marker_config/'
    robot=robot_obj(robot_name,def_path=config_dir+robot_name+'_robot_default_config.yml',tool_file_path=config_dir+tool_name+'.csv',\
    base_marker_config_file=robot_marker_dir+robot_name+'_marker_config.yaml',tool_marker_config_file=tool_marker_dir+tool_name+'_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=phasespace_mocap'
    mocap_cli = RRN.ConnectService(mocap_url)
    
    rob_ip='192.168.1.101'

    calib_obj = CalibRobotPH(mocap_cli,robot)

    # calibration
    ## zero config

    # collecting raw data
    raw_data_dir='PH_rotate_data/train_data'
    # raw_data_dir='PH_raw_data/valid_data_1'
    # raw_data_dir='PH_raw_data/valid_data_2'
    #####################

    calib_obj.run_calib(rob_ip,rob_speed=2,repeat_N=1\
                        ,raw_data_dir=raw_data_dir) # save calib config to file
    print("Collect PH data done")

def calib_R1():

    config_dir='../config/'
    robot_name='M10ia'
    tool_name='ge_R1_tool'
    robot_marker_dir=config_dir+robot_name+'_marker_config/'
    tool_marker_dir=config_dir+tool_name+'_marker_config/'
    robot=robot_obj(robot_name,def_path=config_dir+robot_name+'_robot_default_config.yml',tool_file_path=config_dir+tool_name+'.csv',\
    base_marker_config_file=robot_marker_dir+robot_name+'_marker_config.yaml',tool_marker_config_file=tool_marker_dir+tool_name+'_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=phasespace_mocap'
    mocap_cli = RRN.ConnectService(mocap_url)
    
    rob_ip='192.168.1.101'

    calib_obj = CalibRobotPH(mocap_cli,robot)

    # calibration
    ## zero config

    # collecting raw data
    raw_data_dir='PH_rotate_data/train_data'
    # raw_data_dir='PH_raw_data/valid_data_1'
    # raw_data_dir='PH_raw_data/valid_data_2'
    #####################

    calib_obj.run_calib(rob_ip,rob_speed=2,repeat_N=1\
                        ,raw_data_dir=raw_data_dir) # save calib config to file
    print("Collect PH data done")


if __name__=='__main__':

    calib_R1()
    # calib_R2()