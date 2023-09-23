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

    def run_calib(self,base_marker_config_file,rob_IP=None,ROBOT_CHOICE=None,rob_p2d=None,start_p=None,paths=[],rob_speed=3,repeat_N=1,
                  raw_data_dir=''):
        
        client = MotionProgramExecClient()

        input("Press Enter and the robot will start moving.")
        for j in range(len(paths)-1,-1,-1): # from axis 6 to axis 1
            mp=MotionProgram(ROBOT_CHOICE=ROBOT_CHOICE,pulse2deg=rob_p2d)
            mp.MoveJ(start_p[j],rob_speed,0)
            # mp.MoveJ(paths[j][0],rob_speed,0)
            client.execute_motion_program(mp)
            time.sleep(3)

            mp=MotionProgram(ROBOT_CHOICE=ROBOT_CHOICE,pulse2deg=rob_p2d)
            for N in range(repeat_N):
                mp.MoveJ(paths[j][0],rob_speed,0)
                mp.MoveJ(paths[j][1],rob_speed,0)
            mp.MoveJ(start_p[j],rob_speed,0)
            self.mpl_obj.run_pose_listener()
            robot_stamps,curve_exe, job_line,job_step = client.execute_motion_program(mp)
            self.mpl_obj.stop_pose_listener()
            curve_p,curve_R,timestamps = self.mpl_obj.get_frames_traj()

            with open(raw_data_dir+'_'+str(j+1)+'_mocap_p.pickle', 'wb') as handle:
                pickle.dump(curve_p, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(raw_data_dir+'_'+str(j+1)+'_mocap_R.pickle', 'wb') as handle:
                pickle.dump(curve_R, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(raw_data_dir+'_'+str(j+1)+'_mocap_timestamps.pickle', 'wb') as handle:
                pickle.dump(timestamps, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(raw_data_dir+'_'+str(j+1)+'_robot_q.pickle', 'wb') as handle:
                pickle.dump(curve_exe, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(raw_data_dir+'_'+str(j+1)+'_robot_timestamps.pickle', 'wb') as handle:
                pickle.dump(robot_stamps, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # save zero config
        mp=MotionProgram(ROBOT_CHOICE=ROBOT_CHOICE,pulse2deg=rob_p2d)
        mp.MoveJ(start_p[-1],rob_speed,0)
        client.execute_motion_program(mp)
        
        mp=MotionProgram(ROBOT_CHOICE=ROBOT_CHOICE,pulse2deg=rob_p2d)
        mp.MoveJ(start_p[-1],rob_speed,0)
        mp.setWaitTime(5)
        self.mpl_obj.run_pose_listener()
        robot_stamps,curve_exe, job_line,job_step = client.execute_motion_program(mp)
        self.mpl_obj.stop_pose_listener()
        curve_p,curve_R,timestamps = self.mpl_obj.get_frames_traj()
        
        with open(raw_data_dir+'_zero_mocap_p.pickle', 'wb') as handle:
            pickle.dump(curve_p, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(raw_data_dir+'_zero_mocap_R.pickle', 'wb') as handle:
            pickle.dump(curve_R, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(raw_data_dir+'_zero_mocap_timestamps.pickle', 'wb') as handle:
            pickle.dump(timestamps, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(raw_data_dir+'_zero_robot_q.pickle', 'wb') as handle:
            pickle.dump(curve_exe, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(raw_data_dir+'_zero_robot_timestamps.pickle', 'wb') as handle:
            pickle.dump(robot_stamps, handle, protocol=pickle.HIGHEST_PROTOCOL)

def calib_S1():

    config_dir='../config/'
    turn_table=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv'\
        ,pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
        base_marker_config_file=config_dir+'D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
    mocap_cli = RRN.ConnectService(mocap_url)

    calib_obj = CalibRobotPH(mocap_cli,turn_table)

    # calibration
    ## zero config
    start_p = np.array([[0,0],
                        [0,0]])
    q1_1=start_p[0] + np.array([-30,0])
    q1_2=start_p[0] + np.array([40,0])
    q2_1=start_p[1] + np.array([0,-60])
    q2_2=start_p[1] + np.array([0,60])

    q_paths = [[q1_1,q1_2],[q2_1,q2_2]]
    
    # collecting raw data
    raw_data_dir='PH_rotate_data/train_data'
    # raw_data_dir='PH_raw_data/valid_data_1'
    # raw_data_dir='PH_raw_data/valid_data_2'
    #####################

    calib_obj.run_calib(config_dir+'D500B_robot_default_config.yaml','192.168.1.31','ST1',turn_table.pulse2deg,start_p,q_paths,rob_speed=2,repeat_N=1\
                        ,raw_data_dir=raw_data_dir) # save calib config to file
    print("Collect PH data done")

def calib_R2():

    config_dir='../config/'
    robot=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
	pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA1440_marker_config.yaml',tool_marker_config_file=config_dir+'mti_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
    mocap_cli = RRN.ConnectService(mocap_url)

    calib_obj = CalibRobotPH(mocap_cli,robot)

    # calibration
    ## zero config
    start_p = np.array([[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]])
    q1_1=start_p[0] + np.array([-45,0,0,0,0,0])
    q1_2=start_p[0] + np.array([45,0,0,0,0,0])
    q2_1=start_p[1] + np.array([0,55,0,0,0,0])
    q2_2=start_p[1] + np.array([0,-10,0,0,0,0])
    q3_1=start_p[2] + np.array([0,0,-70,0,0,0])
    q3_2=start_p[2] + np.array([0,0,10,0,0,0])
    q4_1=start_p[3] + np.array([0,0,0,-60,0,0])
    q4_2=start_p[3] + np.array([0,0,0,60,0,0])
    q5_1=start_p[4] + np.array([0,0,0,0,10,0])
    q5_2=start_p[4] + np.array([0,0,0,0,-60,0])
    q6_1=start_p[5] + np.array([0,0,0,0,0,-60])
    q6_2=start_p[5] + np.array([0,0,0,0,0,60])

    q_paths = [[q1_1,q1_2],[q2_1,q2_2],[q3_1,q3_2],[q4_1,q4_2],[q5_1,q5_2],[q6_1,q6_2]]

    # collecting raw data
    raw_data_dir='PH_rotate_data/train_data'
    #####################

    calib_obj.run_calib(config_dir+'MA1440_marker_config.yaml','192.168.1.31','RB2',robot.pulse2deg,start_p,q_paths,rob_speed=1,repeat_N=1\
                        ,raw_data_dir=raw_data_dir) # save calib config to file
    print("Collect PH data done")

def calib_R1():

    config_dir='../config/'
    robot_marker_dir=config_dir+'MA2010_marker_config/'
    tool_marker_dir=config_dir+'weldgun_marker_config/'
    robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
    base_marker_config_file=robot_marker_dir+'MA2010_marker_config.yaml',tool_marker_config_file=tool_marker_dir+'weldgun_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
    mocap_cli = RRN.ConnectService(mocap_url)

    calib_obj = CalibRobotPH(mocap_cli,robot_weld)

    # calibration
    ## zero config
    start_p = np.array([[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]])
    q1_1=start_p[0] + np.array([-45,0,0,0,0,0])
    q1_2=start_p[0] + np.array([45,0,0,0,0,0])
    q2_1=start_p[1] + np.array([0,50,0,0,0,0])
    q2_2=start_p[1] + np.array([0,-10,0,0,0,0])
    q3_1=start_p[2] + np.array([0,0,-60,0,0,0])
    q3_2=start_p[2] + np.array([0,0,10,0,0,0])
    q4_1=start_p[3] + np.array([0,0,0,-90,0,0])
    q4_2=start_p[3] + np.array([0,0,0,90,0,0])
    q5_1=start_p[4] + np.array([0,0,0,0,45,0])
    q5_2=start_p[4] + np.array([0,0,0,0,-80,0])
    q6_1=start_p[5] + np.array([0,0,0,0,0,-120])
    q6_2=start_p[5] + np.array([0,0,0,0,0,120])
    ## out stretch
    # start_p = np.array([[0,50,31,0,0,0],
    #                     [0,50,31,0,0,0],
    #                     [0,50,31,0,0,0],
    #                     [0,50,31,0,0,0],
    #                     [0,50,31,0,0,0],
    #                     [0,50,31,0,0,0]])
    # q1_1=start_p[0] + np.array([-53,0,0,0,0,0])
    # q1_2=start_p[0] + np.array([44,0,0,0,0,0])
    # q2_1=start_p[1] + np.array([0,30,0,0,0,0])
    # q2_2=start_p[1] + np.array([0,-10,0,0,0,0])
    # q3_1=start_p[2] + np.array([0,0,-50,0,0,0])
    # q3_2=start_p[2] + np.array([0,0,50,0,0,0])
    # q4_1=start_p[3] + np.array([0,0,0,-120,0,0])
    # q4_2=start_p[3] + np.array([0,0,0,120,0,0])
    # q5_1=start_p[4] + np.array([0,0,0,0,80,0])
    # q5_2=start_p[4] + np.array([0,0,0,0,-80,0])
    # q6_1=start_p[5] + np.array([0,0,0,0,0,-180])
    # q6_2=start_p[5] + np.array([0,0,0,0,0,180])
    ## inward
    # start_p = np.array([[0,-66,-66,0,0,0],
    #                     [0,-66,-66,0,0,0],
    #                     [0,-66,-66,0,0,0],
    #                     [0,-66,-66,0,0,0],
    #                     [0,-66,-66,0,0,0],
    #                     [0,-66,-66,0,0,0]])
    # q1_1=start_p[0] + np.array([-80,0,0,0,0,0])
    # q1_2=start_p[0] + np.array([60,0,0,0,0,0])
    # q2_1=start_p[1] + np.array([0,70,0,0,0,0])
    # q2_2=start_p[1] + np.array([0,0,0,0,0,0])
    # q3_1=start_p[2] + np.array([0,0,-8,0,0,0])
    # q3_2=start_p[2] + np.array([0,0,30,0,0,0])
    # q4_1=start_p[3] + np.array([0,0,0,-120,0,0])
    # q4_2=start_p[3] + np.array([0,0,0,120,0,0])
    # q5_1=start_p[4] + np.array([0,0,0,0,80,0])
    # q5_2=start_p[4] + np.array([0,0,0,0,-80,0])
    # q6_1=start_p[5] + np.array([0,0,0,0,0,-180])
    # q6_2=start_p[5] + np.array([0,0,0,0,0,180])

    q_paths = [[q1_1,q1_2],[q2_1,q2_2],[q3_1,q3_2],[q4_1,q4_2],[q5_1,q5_2],[q6_1,q6_2]]

    # collecting raw data
    raw_data_dir='PH_rotate_data/train_data'
    # raw_data_dir='PH_raw_data/valid_data_1'
    # raw_data_dir='PH_raw_data/valid_data_2'
    #####################

    calib_obj.run_calib(config_dir+'MA2010_marker_config.yaml','192.168.1.31','RB1',robot_weld.pulse2deg,start_p,q_paths,rob_speed=2,repeat_N=1\
                        ,raw_data_dir=raw_data_dir) # save calib config to file
    print("Collect PH data done")


if __name__=='__main__':

    calib_R1()
    
    # calib_S1()
    # calib_R2()