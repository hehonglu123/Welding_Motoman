from copy import deepcopy
import sys
sys.path.append('toolbox/')
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
        self.base_markers_ids = robot.base_markers_id
        self.base_rigid_id = robot.base_rigid_id

        # mocap listener
        all_ids=[]
        all_ids.extend(self.calib_marker_ids)
        all_ids.extend(robot.calib_markers_id)
        all_ids.extend(self.base_markers_ids)
        all_ids.append(self.base_rigid_id)
        all_ids.append(robot.tool_rigid_id)
        all_ids.append("marker10_rigid0")
        print('Calib ID:',all_ids)
        self.mpl_obj = MocapFrameListener(self.mocap_cli,all_ids,'world')

    def run_calib(self,rob_ip=None,start_p=None,paths=[],rob_speed=3,repeat_N=1,
                  raw_data_dir='',uframe_num=1,utool_num=1,robot_group=1):
        
        client = FANUCClient(rob_ip)
        
        input("Press Enter and the robot will start moving.")
        tp= TPMotionProgram(tool_num=utool_num,uframe_num=uframe_num)
        jtend = jointtarget(robot_group,uframe_num,utool_num,start_p[-1],[0]*6)
        tp.moveJ(jtend,rob_speed,'%',-1)
        tp.setIO('DO',10,False)
        
        client.set_ioport('DOUT',10,True)
        client.execute_motion_program(tp,record_joint=False,non_block=True)
        while True:
            io_res=client.read_ioport('DOUT',10)
            if not io_res:
                break
        
        for j in range(5,-1,-1): # from axis 6 to axis 1
            
            tp= TPMotionProgram(tool_num=utool_num,uframe_num=uframe_num)
            for N in range(repeat_N):
                jt1 = jointtarget(robot_group,uframe_num,utool_num,paths[j][0],[0]*6)
                jt2 = jointtarget(robot_group,uframe_num,utool_num,paths[j][1],[0]*6)  
                tp.moveJ(jt1,rob_speed,'%',-1)
                tp.moveJ(jt2,rob_speed,'%',-1)
            jtend = jointtarget(robot_group,uframe_num,utool_num,start_p[j],[0]*6)
            tp.moveJ(jtend,rob_speed,'%',-1)
            tp.setIO('DO',10,False)

            input("Press Enter and Start Rotating Joint "+str(j+1))
            self.mpl_obj.run_pose_listener()
            print("The robot is moving. Please wait")
            client.set_ioport('DOUT',10,True)
            client.execute_motion_program(tp,record_joint=False,non_block=True)
            while True:
                io_res=client.read_ioport('DOUT',10)
                if not io_res:
                    break
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
        tp= TPMotionProgram(tool_num=utool_num,uframe_num=uframe_num)
        jtend = jointtarget(robot_group,uframe_num,utool_num,start_p[0],[0]*6)
        tp.moveJ(jtend,rob_speed,'%',-1)
        tp.setIO('DO',10,False)
        
        client.set_ioport('DOUT',10,True)
        client.execute_motion_program(tp,record_joint=False,non_block=True)
        while True:
            io_res=client.read_ioport('DOUT',10)
            if not io_res:
                break
        
        print("Reading Zero Config")
        self.mpl_obj.run_pose_listener()
        curve_exe = client.get_joint_angle(read_N=3)
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

    config_dir='config/'
    robot_name='LRMATE200id'
    tool_name='ge_R2_tool'
    robot_marker_dir=config_dir+robot_name+'_marker_config/'
    tool_marker_dir=config_dir+tool_name+'_marker_config/'
    uframe_num=4
    utool_num=1
    robot_group=2

    robot=robot_obj(robot_name,def_path=config_dir+robot_name+'_robot_default_config.yml',tool_file_path=config_dir+tool_name+'.csv',\
    base_marker_config_file=robot_marker_dir+robot_name+'_marker_config.yaml',tool_marker_config_file=tool_marker_dir+tool_name+'_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=phasespace_mocap'
    mocap_cli = RRN.ConnectService(mocap_url)
    
    rob_ip='127.0.0.2'
    rob_speed=3

    calib_obj = CalibRobotPH(mocap_cli,robot)

    # calibration
    ## zero config
    # calibration
    ## zero config
    # calibration
    ## zero config
    start_p = np.repeat([[-25,-30,-50,0,-75,-20]],repeats=6,axis=0)
    q1_1=start_p[0] + np.array([10,0,0,0,0,0])
    q1_2=start_p[0] + np.array([-20,0,0,0,0,0])
    q2_1=start_p[1] + np.array([0,20,0,0,0,0])
    q2_2=start_p[1] + np.array([0,-20,0,0,0,0])
    q3_1=start_p[2] + np.array([0,0,-20,0,0,0])
    q3_2=start_p[2] + np.array([0,0,20,0,0,0])
    q4_1=start_p[3] + np.array([0,0,0,-20,0,0])
    q4_2=start_p[3] + np.array([0,0,0,20,0,0])
    q5_1=start_p[4] + np.array([0,0,0,0,0,0])
    q5_2=start_p[4] + np.array([0,0,0,0,40,0])
    q6_1=start_p[5] + np.array([0,0,0,0,0,10])
    q6_2=start_p[5] + np.array([0,0,0,0,0,-30])
    
    q_paths = [[q1_1,q1_2],[q2_1,q2_2],[q3_1,q3_2],[q4_1,q4_2],[q5_1,q5_2],[q6_1,q6_2]]


    # collecting raw data
    raw_data_dir='PH_rotate_data/train_data'
    # raw_data_dir='PH_raw_data/valid_data_1'
    # raw_data_dir='PH_raw_data/valid_data_2'
    #####################

    calib_obj.run_calib(rob_ip,rob_speed=rob_speed,repeat_N=1,\
                        start_p=start_p,paths=q_paths,raw_data_dir=raw_data_dir,\
                        uframe_num=uframe_num,utool_num=utool_num,robot_group=robot_group) # save calib config to file
    print("Collect PH data done")

def calib_R1():

    config_dir='config/'
    robot_name='M10ia'
    tool_name='ge_R1_tool'
    robot_marker_dir=config_dir+robot_name+'_marker_config/'
    tool_marker_dir=config_dir+tool_name+'_marker_config/'
    uframe_num=4
    utool_num=1
    robot_group=1

    robot=robot_obj(robot_name,def_path=config_dir+robot_name+'_robot_default_config.yml',tool_file_path=config_dir+tool_name+'.csv',\
    base_marker_config_file=robot_marker_dir+robot_name+'_marker_config.yaml',tool_marker_config_file=tool_marker_dir+tool_name+'_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=phasespace_mocap'
    mocap_cli = RRN.ConnectService(mocap_url)
    
    # rob_ip='127.0.0.2'
    rob_ip='192.168.0.1'
    
    
    rob_speed=3

    calib_obj = CalibRobotPH(mocap_cli,robot)

    # calibration
    ## zero config
    # calibration
    ## zero config
    start_p = np.repeat([[-35,-32,-63,1.6,-56,-33]],repeats=6,axis=0)
    
    q1_1=start_p[0] + np.array([35,0,0,0,0,0])
    q1_2=start_p[0] + np.array([-55,0,0,0,0,0])
    q2_1=start_p[1] + np.array([0,17,0,0,0,0])
    q2_2=start_p[1] + np.array([0,-14,0,0,0,0])
    q3_1=start_p[2] + np.array([0,0,-17,0,0,0])
    q3_2=start_p[2] + np.array([0,0,30,0,0,0])
    q4_1=start_p[3] + np.array([0,0,0,-111.6,0,0])
    q4_2=start_p[3] + np.array([0,0,0,98.4,0,0])
    q5_1=start_p[4] + np.array([0,0,0,0,-64,0])
    q5_2=start_p[4] + np.array([0,0,0,0,89,0])
    q6_1=start_p[5] + np.array([0,0,0,0,0,88])
    q6_2=start_p[5] + np.array([0,0,0,0,0,-107])

    q_paths = [[q1_1,q1_2],[q2_1,q2_2],[q3_1,q3_2],[q4_1,q4_2],[q5_1,q5_2],[q6_1,q6_2]]

    # collecting raw data
    raw_data_dir='PH_rotate_data/train_data'
    #####################

    calib_obj.run_calib(rob_ip,rob_speed=rob_speed,repeat_N=1,\
                        start_p=start_p,paths=q_paths,raw_data_dir=raw_data_dir,\
                        uframe_num=uframe_num,utool_num=utool_num,robot_group=robot_group) # save calib config to file
    print("Collect PH data done")


if __name__=='__main__':

    calib_R1()
    # calib_R2()