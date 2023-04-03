from copy import deepcopy
import sys
sys.path.append('../toolbox/')
sys.path.append('../redundancy_resolution/')
from utils import *

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
from threading import Thread
import numpy as np
import time
import yaml
from fitting_3dcircle import fitting_3dcircle

class CalibRobotTool:
    def __init__(self,mocap_cli,tool_markers,tool_marker_ids,tool_rigid_id):

        self.mocap_cli = mocap_cli
        self.tool_markers = tool_markers
        self.tool_marker_ids = tool_marker_ids
        self.tool_rigid_id = tool_rigid_id
        
        self.clear_samples()
        self.sample_threshold = 0.3 # mm

        self.collect_thread_end = True
    
    def clear_samples(self):

        self.marker_position_table = {}
        for i in range(len(self.tool_marker_ids)):
            self.marker_position_table[self.tool_marker_ids[i]]=[]
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
            # make sure we did get tool rigid body
            T_rigid_mocap = None
            for i in range(len(data.fiducials.recognized_fiducials)):
                if data.fiducials.recognized_fiducials[i].fiducial_marker == self.tool_rigid_id:
                    this_position = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['position']))
                    this_orientation = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['orientation']))
                    if last_T_rigid_mocap is None:
                        T_rigid_mocap = Transform(q2R(this_orientation,this_position))
                    else:
                        if np.linalg.norm(this_position-last_T_rigid_mocap.p)>=self.sample_threshold or np.linalg.norm(this_orientation-R2q(last_T_rigid_mocap))>=self.sample_threshold:
                            T_rigid_mocap = Transform(q2R(this_orientation,this_position))
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

        # A: known tool maker position in tool frame
        # B: capture tool marker position in rigid body frame (from motiv)

        # find center of A
        marker_A = deepcopy(self.tool_markers)
        center_A = []
        A = []
        for marker_id in self.tool_marker_ids:
            center_A.append(marker_A[marker_id])
            A.append(marker_A[marker_id])
        center_A = np.mean(center_A,axis=0)
        A = np.array(A)
        # find center of B
        marker_B = {}
        for marker_id in self.tool_marker_ids: # average across the captured
            marker_B[marker_id] = np.mean(self.marker_position_table[marker_id],axis=0)
        center_B = []
        B = []
        for marker_id in self.tool_marker_ids:
            center_B.append(marker_B[marker_id])
            B.append(marker_B[marker_id])
        center_B = np.mean(center_B,axis=0)
        B = np.array(B)
        
        A_centered = A-center_A
        B_centered = B-center_B
        H = np.matmul(A_centered.T,B_centered)
        u,s,v = np.linalg.svd(H)
        R = np.matmul(v,u.T)
        if np.linalg.det(R)<0:
            u,s,v = np.linalg.svd(R)
            v[:,2] = v[:,2]*-1
            R = np.matmul(v,u.T)

        t = center_B-np.dot(R,center_A)
        T_tool_toolmarker = Transform(R,t)

        return T_tool_toolmarker

    def run_calib(self,tool_marker_config_file):

        # 
        if len(self.marker_position_table[self.tool_rigid_id]) == 0:
            tool_thread = Thread( target = self.collect_point_thread)
            self.collect_thread_end = False
            time.sleep(3)
            self.collect_thread_end = True
            tool_thread.join()
        if len(self.marker_position_table[self.tool_rigid_id]) == 0:
            raise("problem!!! No base markers!")

        T_tool_toolmarker = self.find_tool() # T^base_mocap, T^tool_toolmarker

        with open(tool_marker_config_file,'r') as file:
            tool_marker_data = yaml.safe_load(file)
        tool_marker_data['Calib_tool_toolmarker_pose'] = {}
        tool_marker_data['Calib_tool_toolmarker_pose']['position'] = {}
        tool_marker_data['Calib_tool_toolmarker_pose']['position']['x'] = T_tool_toolmarker.p[0]
        tool_marker_data['Calib_tool_toolmarker_pose']['position']['y'] = T_tool_toolmarker.p[0]
        tool_marker_data['Calib_tool_toolmarker_pose']['position']['z'] = T_tool_toolmarker.p[0]
        quat = R2q(T_tool_toolmarker.R)
        tool_marker_data['Calib_tool_toolmarker_pose']['orientation'] = {}
        tool_marker_data['Calib_tool_toolmarker_pose']['orientation']['w'] = quat[0]
        tool_marker_data['Calib_tool_toolmarker_pose']['orientation']['x'] = quat[1]
        tool_marker_data['Calib_tool_toolmarker_pose']['orientation']['y'] = quat[2]
        tool_marker_data['Calib_tool_toolmarker_pose']['orientation']['z'] = quat[3]

        with open(tool_marker_config_file,'w') as file:
            yaml.safe_load(tool_marker_data,file)

if __name__=='__main__':

    config_dir='../config/'
    robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg.csv',\
    base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
    mocap_cli = RRN.ConnectService(mocap_url)

    calib_obj = CalibRobotTool(mocap_cli,robot_weld.tool_markers,robot_weld.tool_markers_id,robot_weld.tool_rigid_id)

    # start calibration
    calib_obj.run_calib()