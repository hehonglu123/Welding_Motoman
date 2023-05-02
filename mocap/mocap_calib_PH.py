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
    def __init__(self,mocap_cli,robot,nominal_robot_base) -> None:
        
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

        # nominal H axis
        self.H_nom = np.matmul(nominal_robot_base.R,self.robot.robot.H)
    
    def detect_axis(self,points,rough_axis_direction):

        all_normals=[]
        all_centers=[]
        for i in range(len(self.calib_marker_ids)):
            center, normal = fitting_3dcircle(points[self.calib_marker_ids[i]])
            if np.sum(np.multiply(normal,rough_axis_direction)) < 0:
                normal = -1*normal
            all_normals.append(normal)
            all_centers.append(center)
        normal_mean = np.mean(all_normals,axis=0)
        normal_mean = normal_mean/np.linalg.norm(normal_mean)
        center_mean = np.mean(all_centers,axis=0)

        return center_mean,normal_mean

    def run_calib(self,base_marker_config_file,rob_IP=None,ROBOT_CHOICE=None,rob_p2d=None,start_p=None,paths=[],rob_speed=3,repeat_N=1,
                  save_raw_data=False,raw_data_dir=''):
        
        client = MotionProgramExecClient()

        self.H_act = deepcopy(self.H_nom)
        self.axis_p = deepcopy(self.H_nom)

        input("Press Enter and the robot will start moving.")
        for j in range(len(self.H_nom[0])-1,-1,-1): # from axis 6 to axis 1
            mp=MotionProgram(ROBOT_CHOICE=ROBOT_CHOICE,pulse2deg=rob_p2d)
            mp.MoveJ(start_p[j],rob_speed,0)
            client.execute_motion_program(mp)

            self.mpl_obj.run_pose_listener()
            mp=MotionProgram(ROBOT_CHOICE=ROBOT_CHOICE,pulse2deg=rob_p2d)
            for N in range(repeat_N):
                mp.MoveJ(paths[j][0],rob_speed,0)
                mp.MoveJ(paths[j][1],rob_speed,0)
            client.execute_motion_program(mp)
            self.mpl_obj.stop_pose_listener()
            curve_p,curve_R,timestamps = self.mpl_obj.get_frames_traj()
            axis_p,axis_normal = self.detect_axis(curve_p,self.H_nom[:,j])
            self.H_act[:,j] = axis_normal
            self.axis_p[:,j] = axis_p

            if save_raw_data:
                with open(raw_data_dir+'_'+str(j+1)+'.pickle', 'wb') as handle:
                    pickle.dump(curve_p, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # save zero config
        mp=MotionProgram(ROBOT_CHOICE=ROBOT_CHOICE,pulse2deg=rob_p2d)
        mp.MoveJ(start_p[-1],rob_speed,0)
        client.execute_motion_program(mp)
        self.mpl_obj.run_pose_listener()
        mp=MotionProgram(ROBOT_CHOICE=ROBOT_CHOICE,pulse2deg=rob_p2d)
        mp.MoveJ(start_p[-1],rob_speed,0)
        mp.setWaitTime(5)
        client.execute_motion_program(mp)
        self.mpl_obj.stop_pose_listener()
        curve_p,curve_R,timestamps = self.mpl_obj.get_frames_traj()
        if save_raw_data:
            with open(raw_data_dir+'_zero_config.pickle', 'wb') as handle:
                pickle.dump(curve_p, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(base_marker_config_file,'r') as file:
            base_marker_data = yaml.safe_load(file)
        base_marker_data['H']=[]
        base_marker_data['H_point']=[]
        for j in range(len(self.H_act[0])):
            this_H = {}
            this_H['x']=float(self.H_act[0,j])
            this_H['y']=float(self.H_act[1,j])
            this_H['z']=float(self.H_act[2,j])
            base_marker_data['H'].append(this_H)
            this_Hp = {}
            this_Hp['x']=float(self.axis_p[0,j])
            this_Hp['y']=float(self.axis_p[1,j])
            this_Hp['z']=float(self.axis_p[2,j])
            base_marker_data['H_point'].append(this_Hp)
        with open(base_marker_config_file,'w') as file:
            yaml.safe_dump(base_marker_data,file)
        print("Result Calibrated H:")
        print(self.H_act)
        print("Done. Please check file:",base_marker_config_file)

def calib_R1():

    config_dir='../config/'
    robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
    mocap_cli = RRN.ConnectService(mocap_url)

    # only R matter
    nominal_robot_base = Transform(np.array([[0,1,0],
                                             [0,0,1],
                                             [1,0,0]]),[0,0,0]) 
    calib_obj = CalibRobotPH(mocap_cli,robot_weld,nominal_robot_base)

    # calibration
    # start_p = np.array([[0,-30,-40,0,0,0],
    #                     [0,0,-34,0,0,0],
    #                     [0,0,0,0,0,0],
    #                     [0,0,0,0,-80,0],
    #                     [0,0,0,0,0,0],
    #                     [0,0,0,0,0,0]])
    start_p = np.array([[0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0],
                        [0,0,0,0,0,0]])
    q1_1=start_p[0] + np.array([-80,0,0,0,0,0])
    q1_2=start_p[0] + np.array([56,0,0,0,0,0])
    # q2_1=start_p[1] + np.array([0,-60,0,0,0,0])
    # q2_2=start_p[1] + np.array([0,30,0,0,0,0])
    q2_1=start_p[1] + np.array([0,50,0,0,0,0])
    q2_2=start_p[1] + np.array([0,-10,0,0,0,0])
    q3_1=start_p[2] + np.array([0,0,-60,0,0,0])
    q3_2=start_p[2] + np.array([0,0,10,0,0,0])
    q4_1=start_p[3] + np.array([0,0,0,-120,0,0])
    q4_2=start_p[3] + np.array([0,0,0,120,0,0])
    q5_1=start_p[4] + np.array([0,0,0,0,80,0])
    q5_2=start_p[4] + np.array([0,0,0,0,-80,0])
    q6_1=start_p[5] + np.array([0,0,0,0,0,-180])
    q6_2=start_p[5] + np.array([0,0,0,0,0,180])
    q_paths = [[q1_1,q1_2],[q2_1,q2_2],[q3_1,q3_2],[q4_1,q4_2],[q5_1,q5_2],[q6_1,q6_2]]


    # collecting raw data
    raw_data_dir='PH_raw_data/train_data'
    # raw_data_dir='PH_raw_data/valid_data_1'
    # raw_data_dir='PH_raw_data/valid_data_2'
    #####################

    calib_obj.run_calib(config_dir+'MA2010_marker_config.yaml','192.168.1.31','RB1',robot_weld.pulse2deg,start_p,q_paths,rob_speed=3,repeat_N=1,
                        save_raw_data=True,raw_data_dir=raw_data_dir) # save calib config to file


if __name__=='__main__':

    calib_R1()