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
from dx200_motion_program_exec_client import *
from MocapPoseListener import *

class CalibRobotBaseRMSE:
    def __init__(self,mocap_cli,robot) -> None:
        
        self.robot=robot
        self.mlp = MocapFrameListener(mocap_cli,self.robot.tool_markers_id)

    def clear_traj(self):
        # create holder for marker traj
        self.traj_mocap = {}
        self.traj_robot = {}
        for marker_id in self.robot.tool_markers_id:
            self.traj_mocap[marker_id]=[]
            self.traj_robot[marker_id]=[]

    def find_base_rmse(self):

        # A: known tool maker position in robot frame, p^marker_robot
        # B: capture tool marker position in mocap world frame, p^marker_mocap
        # Find T^robot_mocap

        marker_B = self.traj_mocap
        marker_A = self.traj_robot
        # find center of A B
        center_A = []
        center_B = []
        A = []
        B = []
        for marker_id in self.robot.tool_markers_id:
            center_A.extend(marker_A[marker_id])
            A.extend(marker_A[marker_id])
            center_B.extend(marker_B[marker_id])
            B.extend(marker_B[marker_id])
        center_A = np.mean(center_A,axis=0)
        A = np.array(A)
        center_B = np.mean(center_B,axis=0)
        B = np.array(B)
        
        A_centered = A-center_A
        B_centered = B-center_B
        H = np.matmul(A_centered.T,B_centered)
        u,s,vT = np.linalg.svd(H)
        R = np.matmul(vT.T,u.T)
        if np.linalg.det(R)<0:
            u,s,v = np.linalg.svd(R)
            v=v.T
            v[:,2] = v[:,2]*-1
            R = np.matmul(v,u.T)

        t = center_B-np.dot(R,center_A)
        T_robot_mocap = Transform(R,t)

        return T_robot_mocap

    def run_calib(self,base_marker_config_file,auto=False,rob_IP=None,ROBOT_CHOICE=None,paths=[],rob_speed=3):

        # init trajectory
        self.clear_traj()

        if auto:
            
            input("Press Enter and the robot will start moving")
            print("Robot is collecting samples")
            for i in range(0,len(paths)):
                client=MotionProgramExecClient(IP=rob_IP,ROBOT_CHOICE=ROBOT_CHOICE,pulse2deg=self.robot.pulse2deg)
                client.MoveJ(paths[i],rob_speed,0)
                robot_stamps,curve_js_exe,_,_ = client.execute_motion_program("AAA.JBI")
                T_tool_robot = self.robot.fwd(curve_js_exe[-1][:6])

                self.mlp.run_pose_listener()
                time.sleep(1)
                self.mlp.stop_pose_listener()
                curve_p_mocap,_,_ = self.mlp.get_frames_traj()

                for marker_id in curve_p_mocap.keys():
                    # add mocap marker position
                    self.traj_mocap[marker_id].append(np.mean(curve_p_mocap[marker_id],axis=0))
                    # add robot fwd marker position
                    p_marker_robot = np.matmul(T_tool_robot.R,self.robot.tool_markers[marker_id])+T_tool_robot.p
                    self.traj_robot[marker_id].append(p_marker_robot)
            
            # save this to files
            for marker_id in self.robot.tool_markers_id:
                np.savetxt('calib_data/robot_'+marker_id+'.csv',self.traj_robot[marker_id],delimiter=',')
                np.savetxt('calib_data/mocap_'+marker_id+'.csv',self.traj_mocap[marker_id],delimiter=',')

            # get rmse results
            T_robot_mocap = self.find_base_rmse()
            print(T_robot_mocap)

            # listen to base marker rigid
            self.mlp = MocapFrameListener(self.mlp.rr_mocap,[self.robot.base_rigid_id])
            self.mlp.run_pose_listener()
            time.sleep(1)
            self.mlp.stop_pose_listener()
            curve_p_mocap,curve_R_mocap,_ = self.mlp.get_frames_traj()
            T_base_basemarker = Transform(curve_R_mocap[self.robot.base_rigid_id][-1],curve_p_mocap[self.robot.base_rigid_id][-1]).inv()*T_robot_mocap

            with open(base_marker_config_file,'r') as file:
                base_marker_data = yaml.safe_load(file)
            base_marker_data['calib_base_basemarker_pose'] = {}
            base_marker_data['calib_base_basemarker_pose']['position'] = {}
            base_marker_data['calib_base_basemarker_pose']['position']['x'] = float(T_base_basemarker.p[0])
            base_marker_data['calib_base_basemarker_pose']['position']['y'] = float(T_base_basemarker.p[1])
            base_marker_data['calib_base_basemarker_pose']['position']['z'] = float(T_base_basemarker.p[2])
            quat = R2q(T_base_basemarker.R)
            base_marker_data['calib_base_basemarker_pose']['orientation'] = {}
            base_marker_data['calib_base_basemarker_pose']['orientation']['w'] = float(quat[0])
            base_marker_data['calib_base_basemarker_pose']['orientation']['x'] = float(quat[1])
            base_marker_data['calib_base_basemarker_pose']['orientation']['y'] = float(quat[2])
            base_marker_data['calib_base_basemarker_pose']['orientation']['z'] = float(quat[3])
            base_marker_data['calib_base_mocap_pose'] = {}
            base_marker_data['calib_base_mocap_pose']['position'] = {}
            base_marker_data['calib_base_mocap_pose']['position']['x'] = float(T_robot_mocap.p[0])
            base_marker_data['calib_base_mocap_pose']['position']['y'] = float(T_robot_mocap.p[1])
            base_marker_data['calib_base_mocap_pose']['position']['z'] = float(T_robot_mocap.p[2])
            quat = R2q(T_robot_mocap.R)
            base_marker_data['calib_base_mocap_pose']['orientation'] = {}
            base_marker_data['calib_base_mocap_pose']['orientation']['w'] = float(quat[0])
            base_marker_data['calib_base_mocap_pose']['orientation']['x'] = float(quat[1])
            base_marker_data['calib_base_mocap_pose']['orientation']['y'] = float(quat[2])
            base_marker_data['calib_base_mocap_pose']['orientation']['z'] = float(quat[3])

            with open(base_marker_config_file,'w') as file:
                yaml.safe_dump(base_marker_data,file)

        else:
            print("only support auto now")
            return

def calib_R1():

    auto = True

    config_dir='../config/'
    robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv'\
    ,base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
    mocap_cli = RRN.ConnectService(mocap_url)

    j1_rough_axis_direction = np.array([0,1,0])
    j2_rough_axis_direction = np.array([1,0,0])
    calib_obj = CalibRobotBaseRMSE(mocap_cli,robot_weld)

    q_paths = [np.array([0,0,0,0,0,0]),
               np.array([0.6,43.5556,43.072,0,-40.3177,0]),
               np.array([0,42.2755,26.7827,0,-25.3142,0]),
               np.array([0,45.9035,20.174,0,-15.0820,0]),
               np.array([-18.4190,42.2912,36.9291,0,-35.4495,0]),
               np.array([-18.419,43.704,22.8302,0,-19.9410,0]),
               np.array([-27.1265,43.704,22.8302,0,-19.9410,0]),
               np.array([-27.1257,39.9072,31.7209,0,-32.6308,0]),
               np.array([-39.3997,28.0451,3.9701,-1.2083,-32.6318,1.4711]),
               np.array([-37.7089,13.4645,-15.5943,0,-32.6328,0]),
               np.array([-28.2455,14.7027,-14.1372,6.9892,-33.2765,-13.7348]),
               np.array([-20.4984852,-12.13932017,-33.36271151,-6.03594783,-33.9064863,13.41600106]),
               np.array([-20.499230685186483,-43.75198020561859,-56.123483536713636,-6.03497018823279,-33.902405730231806,13.405006136893025]),
               np.array([-4.50046751205436,-61.29156275734683,-66.07316016505116,-8.04597515780915,-38.73380203927336,13.400608169000344]),
               np.array([-4.50046751205436,-40.49567755136467,-66.10706025577804,-8.04597515780915,-38.732781896471614,13.398409185054003]),
               np.array([-4.50046751205436,4.831510232494896,-31.3387505386213,-8.04597515780915,-38.73176175366987,13.396210201107664]),
               np.array([-4.50046751205436,29.691483409892143,0.5348680981351233,-8.04597515780915,-38.73074161086814,13.394011217161323]),
               np.array([12.656772804316535,46.48513571741509,34.46572001844648,-8.046952797561007,-38.73686246767858,13.396210201107664]),
               np.array([14.067222453613681,32.21758168594649,24.938538965279076,-6.005640995676985,-43.27241736421547,10.412188985923306]),
               np.array([17.371938783071517,7.337689078275313,-5.7171875231415115,-3.310288199798514,-37.253574833948335,6.137364194236948]),
               np.array([16.601857129940466,17.906519425195363,-14.140104509853892,-9.814525468924183,-25.80349202721642,11.670007803230199]),
               np.array([21.372188686944916,-4.209814329734893,-38.218585617803015,-4.251755280839852,-23.36229030265214,2.7025512700527443]),
               np.array([22.716290546185267,-15.730059570001828,-30.73294336174291,-1.4244211184602584,-42.30328170256229,-1.2732117049312766]),
               np.array([38.13208774376223,-25.599613075200253,-38.197241116234245,10.225134164705452,-45.89112393628255,-22.513197642636285]),
               np.array([55.65461364583905,-29.249062379597504,-40.686386666827865,20.882385099733096,-51.683494764563356,-44.091827108077766]),
               np.array([41.85203687166537,-10.865000823361365,-32.32499206699018,15.170036029614161,-38.031943791676106,-30.63404535647256]),
               np.array([19.6337274912996,29.140553904157944,11.779025968672908,-4.043518013693637,-40.11099482162262,3.3578484860622786]),
               np.array([0,-50,-50,0,0,0])]

    # start calibration
    # find transformation matrix from the base marker rigid body (defined in motiv) to the actual robot base
    calib_obj.run_calib(config_dir+'MA2010_marker_config.yaml',auto,'192.168.1.31','RB1',q_paths,rob_speed=3) # save calib config to file


if __name__=='__main__':

    calib_R1()
    # calib_R2()

