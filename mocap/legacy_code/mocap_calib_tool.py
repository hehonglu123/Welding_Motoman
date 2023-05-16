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

class CalibRobotTool:
    def __init__(self,mocap_cli,tool_markers,tool_marker_ids,tool_rigid_id):

        self.mocap_cli = mocap_cli
        self.tool_markers = tool_markers
        self.tool_marker_ids = tool_marker_ids
        self.tool_rigid_id = tool_rigid_id
        
        self.clear_samples()
        self.sample_threshold = 0.3 # mm

        self.collect_markers = False
        self.collect_thread_end = True
    
    def clear_samples(self):

        self.marker_position_table = {}
        for marker_id in self.tool_marker_ids:
            self.marker_position_table[marker_id]=[]
        self.marker_position_table[self.tool_rigid_id]=[]

    def collect_point_thread(self):
        
        sensor_data_srv = self.mocap_cli.fiducials_sensor_data.Connect(-1)
        while sensor_data_srv.Available>=1:
            sensor_data_srv.ReceivePacket()
        last_T_rigid_mocap = None
        while not self.collect_thread_end:
            try:
                data = sensor_data_srv.ReceivePacketWait(timeout=10)
            except:
                continue
            if self.collect_markers:
                # make sure we did get tool rigid body
                T_rigid_mocap = None
                for i in range(len(data.fiducials.recognized_fiducials)):
                    if data.fiducials.recognized_fiducials[i].fiducial_marker == self.tool_rigid_id:
                        this_position = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['position']))
                        if np.all(this_position == np.array([0.,0.,0.])):
                            continue
                        this_orientation = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['orientation']))
                        # if last_T_rigid_mocap is None:
                        #     T_rigid_mocap = Transform(q2R(this_orientation),this_position)
                        # else:
                        #     if np.linalg.norm(this_position-last_T_rigid_mocap.p)>=self.sample_threshold or np.linalg.norm(this_orientation-R2q(last_T_rigid_mocap.R))>=self.sample_threshold:
                        #         T_rigid_mocap = Transform(q2R(this_orientation),this_position)
                        # last_T_rigid_mocap = deepcopy(T_rigid_mocap)
                        T_rigid_mocap = Transform(q2R(this_orientation),this_position)
                        break
                # if no tool rigid body, then continue
                if T_rigid_mocap is None:
                    continue
                T_mocap_rigid = T_rigid_mocap.inv()
                for i in range(len(data.fiducials.recognized_fiducials)):
                    this_marker_id = data.fiducials.recognized_fiducials[i].fiducial_marker
                    if this_marker_id in self.tool_marker_ids:
                        this_position = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['position']))
                        this_position_local = np.dot(T_mocap_rigid.R,this_position)+T_mocap_rigid.p
                        self.marker_position_table[this_marker_id].append(this_position_local)
        sensor_data_srv.Close()
    
    def find_tool(self):

        # A: known tool maker position in tool frame, p^marker_tool
        # B: capture tool marker position in rigid body frame (from motiv), p^marker_toolmarker
        # Find T^tool_toolmarker

        marker_B = {}
        for marker_id in self.tool_marker_ids: # average across the captured
            marker_B[marker_id] = np.mean(self.marker_position_table[marker_id],axis=0)
        # find center of A B
        marker_A = deepcopy(self.tool_markers)
        center_A = []
        center_B = []
        A = []
        B = []
        for marker_id in self.tool_marker_ids:
            center_A.append(marker_A[marker_id])
            A.append(marker_A[marker_id])
            center_B.append(marker_B[marker_id])
            B.append(marker_B[marker_id])
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
        T_tool_toolmarker = Transform(R,t)

        return T_tool_toolmarker

    def run_calib(self,tool_marker_config_file,auto=False,rob_IP=None,ROBOT_CHOICE=None,rob_p2d=None,paths=[],rob_speed=3):

        input("Enter to start calibrate tool")

        client=MotionProgramExecClient()

        cp_thread = Thread( target = self.collect_point_thread,daemon=True)
        self.collect_thread_end = False
        cp_thread.start()
        if auto:
            print("Robot is collecting samples")
            # move robot to start
            mp=MotionProgram(ROBOT_CHOICE=ROBOT_CHOICE,pulse2deg=rob_p2d)
            mp.MoveJ(paths[0],rob_speed,0)
            client.execute_motion_program(mp)
            # collect data
            self.collect_markers = True
            time.sleep(0.5)
            mp=MotionProgram(ROBOT_CHOICE=ROBOT_CHOICE,pulse2deg=rob_p2d)
            for i in range(1,len(paths)):
                mp.MoveJ(paths[i],rob_speed,0)
            mp.MoveJ(paths[0],rob_speed,0)
            client.execute_motion_program(mp)
        else:
            input("Please move the tool to be seen by as many camera as possible.")
            self.collect_markers = True
            # 
            time.sleep(3)
        
        self.collect_markers = False
        self.collect_thread_end = True
        cp_thread.join()
        for tool_marker_id in self.tool_marker_ids:
            if len(self.marker_position_table[tool_marker_id]) == 0:
                raise("problem!!! No tool markers!")

        T_tool_toolmarker = self.find_tool() # T^base_mocap, T^tool_toolmarker

        with open(tool_marker_config_file,'r') as file:
            tool_marker_data = yaml.safe_load(file)
        tool_marker_data['calib_tool_toolmarker_pose'] = {}
        tool_marker_data['calib_tool_toolmarker_pose']['position'] = {}
        tool_marker_data['calib_tool_toolmarker_pose']['position']['x'] = float(T_tool_toolmarker.p[0])
        tool_marker_data['calib_tool_toolmarker_pose']['position']['y'] = float(T_tool_toolmarker.p[1])
        tool_marker_data['calib_tool_toolmarker_pose']['position']['z'] = float(T_tool_toolmarker.p[2])
        quat = R2q(T_tool_toolmarker.R)
        tool_marker_data['calib_tool_toolmarker_pose']['orientation'] = {}
        tool_marker_data['calib_tool_toolmarker_pose']['orientation']['w'] = float(quat[0])
        tool_marker_data['calib_tool_toolmarker_pose']['orientation']['x'] = float(quat[1])
        tool_marker_data['calib_tool_toolmarker_pose']['orientation']['y'] = float(quat[2])
        tool_marker_data['calib_tool_toolmarker_pose']['orientation']['z'] = float(quat[3])

        with open(tool_marker_config_file,'w') as file:
            yaml.safe_dump(tool_marker_data,file)
        print("Result T_tool_toolmarker:")
        print(np.degrees(R2rpy(T_tool_toolmarker.R)))
        print(T_tool_toolmarker.p)

        print("Done. Please check file:",tool_marker_config_file)

def calib_scanner():

    config_dir='../config/'
    robot=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'scanner_tcp2.csv',\
	pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg.csv',\
    base_marker_config_file=config_dir+'MA1440_marker_config.yaml',tool_marker_config_file=config_dir+'scanner_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
    mocap_cli = RRN.ConnectService(mocap_url)

    calib_obj = CalibRobotTool(mocap_cli,robot.tool_markers,robot.tool_markers_id,robot.tool_rigid_id)

    # start calibration
    calib_obj.run_calib(config_dir+'scanner_marker_config.yaml')

def calib_weldgun():

    auto = True

    config_dir='../config/'
    robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
    mocap_cli = RRN.ConnectService(mocap_url)

    calib_obj = CalibRobotTool(mocap_cli,robot_weld.tool_markers,robot_weld.tool_markers_id,robot_weld.tool_rigid_id)

    q1=np.array([0,-68,-68,0,0,0])
    q2=np.array([-36.6018,12.4119,-12.1251,-43.3579,-45.4297,68.1203])
    q3=np.array([21.0753,-1.8803,-27.3509,13.1122,-25.1173,-25.2466])
    q_paths=[q1,q2,q3]

    # start calibration
    calib_obj.run_calib(config_dir+'weldgun_marker_config.yaml',auto,'192.168.1.31','RB1',robot_weld.pulse2deg,q_paths)

if __name__=='__main__':

    calib_weldgun()
    # calib_scanner()