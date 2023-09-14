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
        all_ids.extend(self.base_markers_ids)
        all_ids.append(self.base_rigid_id)
        all_ids.append(robot.tool_rigid_id)
        all_ids.append("marker10_rigid0")
        print('Calib ID:',all_ids)
        self.mpl_obj = MocapFrameListener(self.mocap_cli,all_ids,'world',use_quat=True)
    
    def run_datacollect_sync(self,rob_IP=None,paths=[],rob_speed=3,waittime=1,
                  raw_data_dir='',split_motion=2,uframe_num=1,utool_num=1,robot_group=1):
        
        client = FANUCClient(rob_IP)
        read_N=1
        
        input("Press Enter and the robot will start moving.")
        
        tp= TPMotionProgram(tool_num=utool_num,uframe_num=uframe_num)
        tp.setIO('DO',10,False)
        for q in paths[0:]:
            jt = jointtarget(robot_group,uframe_num,utool_num,q,[0]*6)
            tp.moveJ(jt,rob_speed,'%',-1) # moveJ does not support coordinated motion
            tp.setIO('DO',10,True)
            tp.waitIO('DO',10,False)
        client.execute_motion_program(tp,record_joint=False,non_block=True)
        
        pose_cnt=0
        base_T=[]
        tool_T=[]
        marker_T={}
        robot_joint=[]
        while True:
            res_io=client.read_ioport('DOUT',10)
            
            if res_io:
                self.mpl_obj.run_pose_listener()
                joint_exe = client.get_joint_angle(read_N=read_N)
                self.mpl_obj.stop_pose_listener()
                mocap_curve_p,mocap_curve_R,mocap_timestamps,curve_cond = self.mpl_obj.get_frames_traj_cond()
                
                # base rigid
                for k in range(len(mocap_curve_p[self.robot.base_rigid_id])):
                    this_stamp_data=np.append(mocap_curve_p[self.robot.base_rigid_id][k],\
                                            mocap_curve_R[self.robot.base_rigid_id][k])
                    this_stamp_data=np.append(this_stamp_data,curve_cond[self.robot.base_rigid_id][k])
                    base_T.append(this_stamp_data)
                
                # tool rigid
                for k in range(len(mocap_curve_p[self.robot.tool_rigid_id])):
                    this_stamp_data=np.append(mocap_curve_p[self.robot.tool_rigid_id][k],\
                                            mocap_curve_R[self.robot.tool_rigid_id][k])
                    this_stamp_data=np.append(this_stamp_data,curve_cond[self.robot.tool_rigid_id][k])
                    tool_T.append(this_stamp_data)
                
                # tool marker
                for mkr in self.robot.tool_markers_id:
                    print(len(mocap_curve_p[mkr]))
                    for k in range(len(mocap_curve_p[mkr])):
                        this_stamp_data=np.append(mocap_curve_p[mkr][k],\
                                                mocap_curve_R[mkr][k])
                        this_stamp_data=np.append(this_stamp_data,curve_cond[mkr][k])
                        if mkr in marker_T.keys():
                            marker_T[mkr].append(this_stamp_data)
                        else:
                            marker_T[mkr]=[this_stamp_data]
                
                # robot q
                robot_joint.extend(joint_exe)
                
                if pose_cnt %50==0:
                    np.savetxt(raw_data_dir+'_robot_q_raw.csv',robot_joint,delimiter=',')
                    np.savetxt(raw_data_dir+'_tool_T_raw.csv',tool_T,delimiter=',')
                    np.savetxt(raw_data_dir+'_base_T_raw.csv',base_T,delimiter=',')
                    with open(raw_data_dir+'_marker_raw.pickle', 'wb') as handle:
                        pickle.dump(marker_T, handle, protocol=pickle.HIGHEST_PROTOCOL)

                print("Q raw num:",len(robot_joint))
                print("Tool T raw num:",len(tool_T))
                print("Base T raw num:",len(base_T))
                print("Collect pose:",pose_cnt+1)
                pose_cnt+=1
                print("================================================")
                
                client.set_ioport('DOUT',10,False)
                
                if pose_cnt==len(paths):
                    break
            else:
                time.sleep(0.1)
                
        # last save
        np.savetxt(raw_data_dir+'_robot_q_raw.csv',robot_joint,delimiter=',')
        np.savetxt(raw_data_dir+'_tool_T_raw.csv',tool_T,delimiter=',')
        np.savetxt(raw_data_dir+'_base_T_raw.csv',base_T,delimiter=',')
        with open(raw_data_dir+'_marker_raw.pickle', 'wb') as handle:
            pickle.dump(marker_T, handle, protocol=pickle.HIGHEST_PROTOCOL)

def calib_R2():

    config_dir='config/'
    robot_name='LRMATE200id'
    tool_name='ge_R2_tool'
    robot_marker_dir=config_dir+robot_name+'_marker_config/'
    tool_marker_dir=config_dir+tool_name+'_marker_config/'
    uframe_num=4
    utool_num=1
    robot_group=2

    dataset_date = ''
    print("Dataset Date:",dataset_date)
    
    if dataset_date=='':
        robot=robot_obj(robot_name,def_path=config_dir+robot_name+'_robot_default_config.yml',tool_file_path=config_dir+tool_name+'.csv',\
    base_marker_config_file=robot_marker_dir+robot_name+'_marker_config.yaml',tool_marker_config_file=tool_marker_dir+tool_name+'_marker_config.yaml')
    else:
        robot=robot_obj(robot_name,def_path=config_dir+robot_name+'_robot_default_config.yml',tool_file_path=config_dir+tool_name+'.csv',\
        base_marker_config_file=robot_marker_dir+robot_name+'_'+dataset_date+'_marker_config.yaml',tool_marker_config_file=tool_marker_dir+tool_name+'_'+dataset_date+'_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=phasespace_mocap'
    mocap_cli = RRN.ConnectService(mocap_url)
    
    rob_ip='127.0.0.1'

    calib_obj = CalibRobotPH(mocap_cli,robot)

    # calibration
    q2_up=-40
    q2_low=-40
    q3_up_sample = np.array([[-40,-45],[-20,10],[10,0]]) #[[q2 q3]]
    q3_low_sample = np.array([[-40,-70],[-20,-65],[10,-60]]) #[[q2 q3]]
    d_angle = 5 # 5 degree
    dq_sample = [[0,0,0,0,0,0],\
          [1,0,0,-0,-0,0],[0,1,0,0,0,0],\
          [0,0,1,0,0,0],[0,0,0,1,0,0],\
          [0,0,0,0,1,0],[0,0,0,0,0,1]]
    scale=1
    dq_sample = np.array(dq_sample)*scale

    target_q_zero = np.array([1,0,0,1,1,1])

    # speed
    rob_speed=10
    waittime=0.5 # stop 0.5 sec for sync

    q_paths = []
    forward=True
    for q2 in np.append(np.arange(q2_low,q2_up,d_angle),q2_up):
        q3_low = np.interp(q2,q3_low_sample[:,0],q3_low_sample[:,1])
        q3_up = np.interp(q2,q3_up_sample[:,0],q3_up_sample[:,1])
        this_q_paths=[]
        for q3 in np.append(np.arange(q3_low,q3_up,d_angle),q3_up):
            target_q = deepcopy(target_q_zero)
            target_q[1]=q2
            target_q[2]=q3
            q_path_pose=[]
            for dq in dq_sample:
                q_path_pose.append(target_q+dq)
            if forward:
                this_q_paths.extend(q_path_pose)
            else:
                this_q_paths.extend(q_path_pose[::-1])
        if forward:
            q_paths.extend(this_q_paths)
        else:
            q_paths.extend(this_q_paths[::-1])
        forward = not forward
    print("total pose:",len(q_paths))
    print("Data Base:",dataset_date)
    
    # collecting raw data
    raw_data_dir='PH_grad_data/train_data'
    #####################

    calib_obj.run_datacollect_sync(rob_ip,q_paths,rob_speed=rob_speed,waittime=waittime\
                        ,raw_data_dir=raw_data_dir,uframe_num=uframe_num,utool_num=utool_num,robot_group=robot_group) # save calib config to file
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

    dataset_date = ''
    print("Dataset Date:",dataset_date)
    
    if dataset_date=='':
        robot=robot_obj(robot_name,def_path=config_dir+robot_name+'_robot_default_config.yml',tool_file_path=config_dir+tool_name+'.csv',\
    base_marker_config_file=robot_marker_dir+robot_name+'_marker_config.yaml',tool_marker_config_file=tool_marker_dir+tool_name+'_marker_config.yaml')
    else:
        robot=robot_obj(robot_name,def_path=config_dir+robot_name+'_robot_default_config.yml',tool_file_path=config_dir+tool_name+'.csv',\
        base_marker_config_file=robot_marker_dir+robot_name+'_'+dataset_date+'_marker_config.yaml',tool_marker_config_file=tool_marker_dir+tool_name+'_'+dataset_date+'_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=phasespace_mocap'
    mocap_cli = RRN.ConnectService(mocap_url)
    
    # rob_ip='127.0.0.1'
    rob_ip='192.168.0.1'

    calib_obj = CalibRobotPH(mocap_cli,robot)

    # calibration
    q2_up=-20
    q2_low=-50
    q3_up_sample = np.array([[-50,-45],[-35,-30],[-20,-20]]) #[[q2 q3]]
    q3_low_sample = np.array([[-50,-80],[-35,-75],[-20,-70]]) #[[q2 q3]]
    d_angle = 5 # 5 degree
    dq_sample = [[0,0,0,0,0,0],\
          [1,0,0,-0,-0,0],[0,1,0,0,0,0],\
          [0,0,1,0,0,0],[0,0,0,1,0,0],\
          [0,0,0,0,1,0],[0,0,0,0,0,1]]
    scale=1
    dq_sample = np.array(dq_sample)*scale

    target_q_zero = np.array([-35,0,0,1.6,-56,-33])

    # speed
    rob_speed=10
    waittime=0.5 # stop 0.5 sec for sync

    q_paths = []
    forward=True
    for q2 in np.append(np.arange(q2_low,q2_up,d_angle),q2_up):
        q3_low = np.interp(q2,q3_low_sample[:,0],q3_low_sample[:,1])
        q3_up = np.interp(q2,q3_up_sample[:,0],q3_up_sample[:,1])
        this_q_paths=[]
        for q3 in np.append(np.arange(q3_low,q3_up,d_angle),q3_up):
            target_q = deepcopy(target_q_zero)
            target_q[1]=q2
            target_q[2]=q3
            q_path_pose=[]
            for dq in dq_sample:
                q_path_pose.append(target_q+dq)
            if forward:
                this_q_paths.extend(q_path_pose)
            else:
                this_q_paths.extend(q_path_pose[::-1])
        if forward:
            q_paths.extend(this_q_paths)
        else:
            q_paths.extend(this_q_paths[::-1])
        forward = not forward
    print("total pose:",len(q_paths))
    print("Data Base:",dataset_date)
    # exit()

    # collecting raw data
    raw_data_dir='PH_grad_data/train_data'
    #####################

    calib_obj.run_datacollect_sync(rob_ip,q_paths,rob_speed=rob_speed,waittime=waittime\
                        ,raw_data_dir=raw_data_dir,uframe_num=uframe_num,utool_num=utool_num,robot_group=robot_group) # save calib config to file
    print("Collect PH data done")


if __name__=='__main__':

    calib_R1()
    # calib_R2()
    # calib_S1()