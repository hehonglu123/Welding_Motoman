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
        self.mpl_obj = MocapFrameListener(self.mocap_cli,all_ids,'world',use_quat=True)

    def run_datacollect(self,base_marker_config_file,rob_IP=None,ROBOT_CHOICE=None,rob_p2d=None,paths=[],rob_speed=3,waittime=1,
                  raw_data_dir='',split_motion=2):

        input("Press Enter and the robot will start moving.")
        robot_client = MotionProgramExecClient()

        split_bp = np.arange(0,len(paths),len(paths)/split_motion).astype(int)
        split_bp = np.append(split_bp,len(paths))

        all_mocap_p = {}
        all_mocap_R = {}
        all_mocap_stamp = {}
        all_robot_q = []
        all_robot_stamp = []
        for i in range(split_motion):
            mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=self.robot.pulse2deg)
            for test_q in paths[split_bp[i]:split_bp[i+1]]:
                # move robot
                mp.MoveJ(test_q,rob_speed,0)
                mp.setWaitTime(waittime)

            # Run
            self.mpl_obj.run_pose_listener()
            robot_stamps,curve_exe, job_line,job_step = robot_client.execute_motion_program(mp)
            self.mpl_obj.stop_pose_listener()

            curve_p,curve_R,timestamps = self.mpl_obj.get_frames_traj()

            if len(all_robot_q) ==0:
                all_mocap_p = deepcopy(curve_p)
                all_mocap_R = deepcopy(curve_R)
                all_mocap_stamp = deepcopy(timestamps)
                all_robot_q = deepcopy(curve_exe)
                all_robot_stamp = deepcopy(robot_stamps)
            else:
                for key in curve_p.keys():
                    all_mocap_p[key] = np.vstack((all_mocap_p[key],deepcopy(curve_p[key])))
                    all_mocap_R[key] = np.vstack((all_mocap_R[key],deepcopy(curve_R[key])))
                    all_mocap_stamp[key] = np.append(all_mocap_stamp[key],deepcopy(timestamps[key]))
                all_robot_q = np.vstack((all_robot_q,deepcopy(curve_exe)))
                all_robot_stamp = np.append(all_robot_stamp,deepcopy(robot_stamps))    
            
            for key in curve_p.keys():
                print("key:",key)
                print("mp p:",len(all_mocap_p[key]))
                print("mp R:",len(all_mocap_R[key]))
                print("mp t:",len(all_mocap_stamp[key]))
            print("rob q:",len(all_robot_q))
            print("rob t:",len(all_robot_stamp))

            save_curve_R = {}
            save_curve_R[self.robot.base_rigid_id]=deepcopy(all_mocap_R[self.robot.base_rigid_id])
            save_curve_R[self.robot.tool_rigid_id]=deepcopy(all_mocap_R[self.robot.tool_rigid_id])
            print(save_curve_R[self.robot.base_rigid_id][0])
            print(save_curve_R[self.robot.tool_rigid_id][0])

            # exit()

            with open(raw_data_dir+'_mocap_p_cont.pickle', 'wb') as handle:
                pickle.dump(all_mocap_p, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(raw_data_dir+'_mocap_R_cont.pickle', 'wb') as handle:
                pickle.dump(save_curve_R, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(raw_data_dir+'_robot_q_cont.pickle', 'wb') as handle:
                pickle.dump(all_robot_q, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open(raw_data_dir+'_mocap_timestamps_cont.pickle', 'wb') as handle:
                pickle.dump(all_mocap_stamp, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(raw_data_dir+'_robot_timestamps_cont.pickle', 'wb') as handle:
                pickle.dump(all_robot_stamp, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def run_datacollect_sync(self,rob_IP=None,ROBOT_CHOICE=None,rob_p2d=None,paths=[],rob_speed=3,waittime=1,
                  raw_data_dir='',split_motion=2):
        
        input("Press Enter and the robot will start moving.")
        robot_client = MotionProgramExecClient()

        mp=MotionProgram(ROBOT_CHOICE=ROBOT_CHOICE,pulse2deg=self.robot.pulse2deg)
        start_q = paths[0]+np.array([1,1,1,1,1,1])
        mp.MoveJ(start_q,rob_speed,0)
        robot_client.execute_motion_program(mp)

        mp=MotionProgram(ROBOT_CHOICE=ROBOT_CHOICE,pulse2deg=self.robot.pulse2deg)
        for test_q in paths:
            # move robot
            mp.MoveJ(test_q,rob_speed,0)
            mp.setWaitTime(waittime)
        robot_client.execute_motion_program_nonblocking(mp)
        
        ###streaming
        robot_client.StartStreaming()
        start_time=time.time()

        program_start=False
        state_flag=0
        robot_q_align=[]
        mocap_T_align=[]

        robot_q_raw=[]
        tool_T_raw=[]
        base_T_raw=[]
        
        joint_recording=[]
        robot_stamps=[]
        r_pulse2deg = self.robot.pulse2deg
        T_base_basemarker = self.robot.T_base_basemarker
        T_basemarker_base = T_base_basemarker.inv()
        while True:
            if state_flag & 0x08 == 0 and time.time()-start_time>1.:
                break
            res, data = robot_client.receive_from_robot(0.01)
            if res:
                state_flag=data[16]
                if data[18]==0:
                    program_start=True
                if data[18]!=0 and data[18]%2==0 and program_start: # when the robot stop
                    if len(joint_recording)==0:
                        print("Start collect")
                        self.mpl_obj.run_pose_listener()
                    if ROBOT_CHOICE=='RB1':
                        joint_angle=np.radians(np.divide(np.array(data[20:26]),r_pulse2deg))
                    elif ROBOT_CHOICE=='RB2':
                        joint_angle=np.radians(np.divide(np.array(data[26:32]),r_pulse2deg))
                    joint_recording.append(joint_angle)
                    timestamp=data[0]+data[1]*1e-9
                    robot_stamps.append(timestamp)
                else:
                    if len(joint_recording)>0:
                        robot_stamps=np.array(robot_stamps)
                        joint_recording=np.array(joint_recording)
                        self.mpl_obj.stop_pose_listener()
                        mocap_curve_p,mocap_curve_R,mocap_timestamps = self.mpl_obj.get_frames_traj()

                        print("# of Mocap Data:",len(mocap_timestamps[self.robot.base_rigid_id]))
                        print("# of Robot Data:",len(robot_stamps))

                        start_i = np.argmin(np.fabs(mocap_timestamps[self.robot.base_rigid_id]-(mocap_timestamps[self.robot.base_rigid_id][0]+waittime/5)))
                        end_i = np.argmin(np.fabs(mocap_timestamps[self.robot.base_rigid_id]-(mocap_timestamps[self.robot.base_rigid_id][0]+waittime/5*4)))
                        print("# of Mocap Data Used:",end_i-start_i)
                        this_mocap_ori = []
                        this_mocap_p = []
                        base_rigid_R=mocap_curve_R[self.robot.base_rigid_id]
                        mocap_R=mocap_curve_R[self.robot.tool_rigid_id]
                        base_rigid_p=mocap_curve_p[self.robot.base_rigid_id]
                        mocap_p=mocap_curve_p[self.robot.tool_rigid_id]
                        for k in range(start_i,end_i):
                            tool_T_raw.append(np.append(mocap_p[k],mocap_R[k]))
                            base_T_raw.append(np.append(base_rigid_p[k],base_rigid_R[k]))

                            T_mocap_basemarker = Transform(q2R(base_rigid_R[k]),base_rigid_p[k]).inv()
                            T_marker_mocap = Transform(q2R(mocap_R[k]),mocap_p[k])
                            T_marker_basemarker = T_mocap_basemarker*T_marker_mocap
                            T_marker_base = T_basemarker_base*T_marker_basemarker
                            this_mocap_ori.append(R2rpy(T_marker_base.R))
                            this_mocap_p.append(T_marker_base.p)
                        this_mocap_p = np.mean(this_mocap_p,axis=0)
                        this_mocap_ori = R2q(rpy2R(np.mean(this_mocap_ori,axis=0)))
                        mocap_T_align.append(np.append(this_mocap_p,this_mocap_ori))

                        start_i = np.argmin(np.fabs(robot_stamps-(robot_stamps[0]+waittime/5)))
                        end_i = np.argmin(np.fabs(robot_stamps-(robot_stamps[0]+waittime/5*4)))
                        print("# of Robot Data Used:",end_i-start_i)
                        joint_recording = joint_recording[start_i:end_i]
                        robot_stamps = robot_stamps[start_i:end_i]
                        robot_q_align.append(np.mean(joint_recording,axis=0))
                        print(np.degrees(np.mean(joint_recording,axis=0)))
                        joint_recording=[]
                        robot_stamps=[]
                        self.mpl_obj.clear_traj()

                        print("Q align num:",len(robot_q_align))
                        print("mocap align num:",len(mocap_T_align))
                        print("mocap tool raw num:",len(tool_T_raw))
                        print("mocap base raw num:",len(base_T_raw))
                        print("=========================")
                
        robot_client.servoMH(False)

        np.savetxt(raw_data_dir+'_robot_q_align.csv',robot_q_align,delimiter=',')
        np.savetxt(raw_data_dir+'_mocap_T_align.csv',mocap_T_align,delimiter=',')
        np.savetxt(raw_data_dir+'_tool_T_raw.csv',tool_T_raw,delimiter=',')
        np.savetxt(raw_data_dir+'_base_T_raw.csv',base_T_raw,delimiter=',')

        print("Q align num:",len(robot_q_align))
        print("mocap align num:",len(mocap_T_align))
        print("Tool T raw num:",len(tool_T_raw))
        print("Base T raw num:",len(base_T_raw))

def calib_R2():

    dataset_date = '0804'
    print("Dataset Date:",dataset_date)

    config_dir='../config/'
    robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
	pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA1440_'+dataset_date+'_marker_config.yaml',tool_marker_config_file=config_dir+'mti_'+dataset_date+'_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
    mocap_cli = RRN.ConnectService(mocap_url)

    calib_obj = CalibRobotPH(mocap_cli,robot_scan)

    # calibration
    q2_up=50
    q2_low=-55
    q3_up_sample = np.array([[-55,-40],[0,10],[50,60]]) #[[q2 q3]]
    q3_low_sample = np.array([[-55,-70],[0,-60],[50,0]]) #[[q2 q3]]
    d_angle = 5 # 5 degree
    # add 7 points (at least 6 is needed)
    # dq_sample = [[0,0,0,0,0,0],\
    #       [-1,0,0,0,0,0],[1,0,0,0,0,0],\
    #       [0,-1,-1,0,0,0],[0,-1,1,0,0,0],\
    #       [0,1,1,0,0,0],[0,1,-1,0,0,0]]
    # dq_sample = [[0,0,0,0,0,0],\
    #       [-9,0,0,-9,-9,9],[-6,0,0,-6,-6,6],\
    #       [-3,0,0,-3,-3,3],[4,0,0,4,4,-4],\
    #       [8,0,0,8,8,-8],[12,0,0,12,12,-12]]
    # dq_sample = [[0,0,0,0,0,0],\
    #       [-3,0,0,-3,-3,3],[-2,0,0,-2,-2,2],\
    #       [-1,0,0,-1,-1,1],[1,0,0,1,1,-1],\
    #       [2,0,0,2,2,-2],[3,0,0,3,3,-3]]
    dq_sample = [[0,0,0,0,0,0],\
          [1,0,0,-0,-0,0],[0,1,0,0,0,0],\
          [0,0,1,0,0,0],[0,0,0,1,0,0],\
          [0,0,0,0,1,0],[0,0,0,0,0,1]]
    scale=1
    dq_sample = np.array(dq_sample)*scale

    target_q_zero = np.array([1,0,0,1,1,1])

    # speed
    rob_speed=0.2
    waittime=0.75 # stop 0.5 sec for sync

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

    calib_obj.run_datacollect_sync('192.168.1.31','RB2',robot_scan.pulse2deg,q_paths,rob_speed=rob_speed,waittime=waittime\
                        ,raw_data_dir=raw_data_dir) # save calib config to file
    print("Collect PH data done")

def calib_R1():

    dataset_date = '0801'
    print("Dataset Date:",dataset_date)

    config_dir='../config/'
    robot_marker_dir=config_dir+'MA2010_marker_config/'
    tool_marker_dir=config_dir+'weldgun_marker_config/'
    robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
    base_marker_config_file=robot_marker_dir+'MA2010_'+dataset_date+'_marker_config.yaml',tool_marker_config_file=tool_marker_dir+'weldgun_'+dataset_date+'_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
    mocap_cli = RRN.ConnectService(mocap_url)

    calib_obj = CalibRobotPH(mocap_cli,robot_weld)

    # calibration
    q2_up=50
    q2_low=-55
    q3_up_sample = np.array([[-55,-40],[0,10],[50,50]]) #[[q2 q3]]
    q3_low_sample = np.array([[-55,-70],[0,-50],[50,0]]) #[[q2 q3]]
    d_angle = 5 # 5 degree
    # add 7 points (at least 6 is needed)
    # dq_sample = [[0,0,0,0,0,0],\
    #       [-1,0,0,0,0,0],[1,0,0,0,0,0],\
    #       [0,-1,-1,0,0,0],[0,-1,1,0,0,0],\
    #       [0,1,1,0,0,0],[0,1,-1,0,0,0]]
    # dq_sample = [[0,0,0,0,0,0],\
    #       [-9,0,0,-9,-9,9],[-6,0,0,-6,-6,6],\
    #       [-3,0,0,-3,-3,3],[4,0,0,4,4,-4],\
    #       [8,0,0,8,8,-8],[12,0,0,12,12,-12]]
    # dq_sample = [[0,0,0,0,0,0],\
    #       [-3,0,0,-3,-3,3],[-2,0,0,-2,-2,2],\
    #       [-1,0,0,-1,-1,1],[1,0,0,1,1,-1],\
    #       [2,0,0,2,2,-2],[3,0,0,3,3,-3]]
    dq_sample = [[0,0,0,0,0,0],\
          [1,0,0,-0,-0,0],[0,1,0,0,0,0],\
          [0,0,1,0,0,0],[0,0,0,1,0,0],\
          [0,0,0,0,1,0],[0,0,0,0,0,1]]
    scale=1
    dq_sample = np.array(dq_sample)*scale

    target_q_zero = np.array([1,0,0,1,1,1])

    # speed
    rob_speed=10
    waittime=0.1 # stop 0.5 sec for sync

    # qdummy={-55:-55,-40:-40,-25:-25,-10:-10,15:15,30:25,40:35,50:45}

    q_paths = []
    forward=True
    for q2 in np.append(np.arange(q2_low,q2_up,d_angle),q2_up):
    # for q2 in qdummy.keys():
        q3_low = np.interp(q2,q3_low_sample[:,0],q3_low_sample[:,1])
        q3_up = np.interp(q2,q3_up_sample[:,0],q3_up_sample[:,1])
        this_q_paths=[]
        for q3 in np.append(np.arange(q3_low,q3_up,d_angle),q3_up):
        # for q3 in [qdummy[q2]]:
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

    calib_obj.run_datacollect_sync('192.168.1.31','RB1',robot_weld.pulse2deg,q_paths,rob_speed=rob_speed,waittime=waittime\
                        ,raw_data_dir=raw_data_dir) # save calib config to file
    print("Collect PH data done")


if __name__=='__main__':

    calib_R1()
    # calib_R2()
    # calib_S1()