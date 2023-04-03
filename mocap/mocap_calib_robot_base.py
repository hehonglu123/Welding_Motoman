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

class CalibRobotBase:
    def __init__(self,mocap_cli,calib_marker_ids,base_marker_ids,base_rigid_id,j1_rough_axis_direction,j2_rough_axis_direction):

        # self.mocap_url = mocap_url
        self.mocap_cli = mocap_cli
        
        self.calib_marker_ids = calib_marker_ids
        self.base_markers_ids = base_marker_ids
        self.base_rigid_id = base_rigid_id
        self.clear_samples()
        self.sample_threshold = 0.3 # mm

        self.j1_rough_axis_direction=j1_rough_axis_direction
        self.j2_rough_axis_direction=j2_rough_axis_direction

        self.collect_thread_end = True
    
    def clear_samples(self):

        self.marker_position_table = {}
        self.marker_orientation_table = {}
        for i in range(len(self.calib_marker_ids)):
            self.marker_position_table[self.calib_marker_ids[i]]=[]
            self.marker_orientation_table[self.calib_marker_ids[i]]=[]
        for i in range(len(self.base_markers_ids)):
            self.marker_position_table[self.base_markers_ids[i]]=[]
            self.marker_orientation_table[self.base_markers_ids[i]]=[]

    def collect_point_thread(self):
        
        sensor_data_srv = self.mocap_cli.fiducials_sensor_data.Connect(-1)
        while sensor_data_srv.Available>=1:
            sensor_data_srv.ReceivePacket()
        while not self.collect_thread_end:
            try:
                data = sensor_data_srv.ReceivePacketWait(timeout=10)
            except:
                continue
            for i in range(len(data.fiducials.recognized_fiducials)):
                this_marker_id = data.fiducials.recognized_fiducials[i].fiducial_marker
                if this_marker_id in self.calib_marker_ids or this_marker_id in self.base_marker_ids or this_marker_id==self.base_rigid_id:
                    this_position = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['position']))
                    this_orientation = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['orientation']))
                    if len(self.marker_position_table[this_marker_id]) == 0:
                        self.marker_position_table[this_marker_id].append(this_position)
                        self.marker_orientation_table[this_marker_id].append(this_orientation)
                    else:
                        last_position = np.array(list(self.marker_position_table[this_marker_id][-1]))
                        last_orientation = np.array(list(self.marker_orientation_table[this_marker_id][-1]))
                        if np.linalg.norm(this_position-last_position)>=self.sample_threshold or np.linalg.norm(this_orientation-last_orientation)>=self.sample_threshold:
                            self.marker_position_table[this_marker_id].append(this_position)
                            self.marker_orientation_table[this_marker_id].append(this_orientation)
        sensor_data_srv.Close()
        # print("end thread")
    def detect_axis(self,rough_axis_direction):

        all_normals=[]
        all_centers=[]
        for i in range(len(self.calib_marker_ids)):
            center, normal = fitting_3dcircle(self.marker_position_table[self.calib_marker_ids[i]])
            if np.sum(np.multiply(normal,rough_axis_direction)) < 0:
                normal = -1*normal
            all_normals.append(normal)
            all_centers.append(center)
        normal_mean = np.mean(all_normals,axis=0)
        normal_mean = normal_mean/np.linalg.norm(normal_mean)
        center_mean = np.mean(all_centers,axis=0)

        return center_mean,normal_mean
    
    def find_base(self,j1_p,j1_normal,j2_p,j2_normal):

        ab_coefficient = np.matmul(np.linalg.pinv(np.array([j1_normal,-j2_normal]).T),
                                    -(j1_p-j2_p))
        j1_center = j1_p+ab_coefficient[0]*j1_normal
        j2_center = j2_p+ab_coefficient[1]*j2_normal
        z_axis = j1_normal
        x_axis = j2_center-j1_center
        x_axis = x_axis-np.dot(x_axis,z_axis)*z_axis
        x_axis = x_axis/np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis,x_axis)
        y_axis = y_axis/np.linalg.norm(y_axis)

        T_base_mocap = Transform(np.array([x_axis,y_axis,z_axis]).T,j1_center)

        T_basemarker_mocap = Transform(q2R(self.marker_orientation_table[self.base_rigid_id][0]),self.self.marker_position_table[self.base_rigid_id][0])

        T_base_basemarker = T_basemarker_mocap.inv()*T_base_mocap

        return T_base_mocap,T_base_basemarker

    def run_calib(self,base_marker_config_file):

        # check where's joint axis 1
        j1_thread = Thread( target = self.collect_point_thread)
        self.collect_thread_end = False
        input("Press Enter and start moving J1")
        time.sleep(0.5)
        j1_thread.start()
        input("Press Enter if you collect enough samples")
        time.sleep(0.5)
        self.collect_thread_end = True
        j1_thread.join()
        for i in range(len(self.calib_marker_ids)):
            if len(self.marker_position_table[self.calib_marker_ids[i]]) == 0:
                raise("problem!!! No J1 Calib markers!")
        print("total sample",len(self.marker_position_table[self.calib_marker_ids[0]]))
        j1_p,j1_normal = self.detect_axis(self.j1_rough_axis_direction)

        self.clear_samples()
        if len(self.marker_position_table[self.calib_marker_ids[0]]) != 0:
            raise("problem!!! Did not clear J1 Calib markers")
        # check where's joint axis 2
        j2_thread = Thread( target = self.collect_point_thread)
        self.collect_thread_end = False
        input("Press Enter and start moving J2")
        time.sleep(0.5)
        j2_thread.start()
        input("Press Enter if you collect enough samples")
        time.sleep(0.5)
        self.collect_thread_end = True
        j2_thread.join()
        for i in range(len(self.calib_marker_ids)):
            if len(self.marker_position_table[self.calib_marker_ids[i]]) == 0:
                raise("problem!!! No J2 Calib markers!")
        print("total sample",len(self.marker_position_table[self.calib_marker_ids[0]]))
        j2_p,j2_normal = self.detect_axis(self.j2_rough_axis_direction)

        if len(self.marker_position_table[self.base_rigid_id]) == 0:
            base_thread = Thread( target = self.collect_point_thread)
            self.collect_thread_end = False
            time.sleep(3)
            self.collect_thread_end = True
            base_thread.join()
        if len(self.marker_position_table[self.base_rigid_id]) == 0:
            raise("problem!!! No base rigid!")

        T_base_mocap,T_base_basemarker = self.find_base(j1_p,j1_normal,j2_p,j2_normal) # T^base_mocap, T^base_basemarker
        T_base_H = H_from_RT(T_base_basemarker.R,T_base_basemarker.p)

        with open(base_marker_config_file,'r') as file:
            base_marker_data = yaml.safe_load(file)
        base_marker_data['Calib_base_basemarker_pose'] = {}
        base_marker_data['Calib_base_basemarker_pose']['position'] = {}
        base_marker_data['Calib_base_basemarker_pose']['position']['x'] = T_base_basemarker.p[0]
        base_marker_data['Calib_base_basemarker_pose']['position']['y'] = T_base_basemarker.p[0]
        base_marker_data['Calib_base_basemarker_pose']['position']['z'] = T_base_basemarker.p[0]
        quat = R2q(T_base_basemarker.R)
        base_marker_data['Calib_base_basemarker_pose']['orientation'] = {}
        base_marker_data['Calib_base_basemarker_pose']['orientation']['w'] = quat[0]
        base_marker_data['Calib_base_basemarker_pose']['orientation']['x'] = quat[1]
        base_marker_data['Calib_base_basemarker_pose']['orientation']['y'] = quat[2]
        base_marker_data['Calib_base_basemarker_pose']['orientation']['z'] = quat[3]

        with open(base_marker_config_file,'w') as file:
            yaml.safe_load(base_marker_data,file)

if __name__=='__main__':

    config_dir='../config/'
    robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg.csv',base_marker_config_file=config_dir+'MA2010_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
    mocap_cli = RRN.ConnectService(mocap_url)

    j1_rough_axis_direction = np.array([0,1,0])
    j2_rough_axis_direction = np.array([1,0,0])
    calib_obj = CalibRobotBase(mocap_cli,robot_weld.calib_markers_id,robot_weld.base_markers_id,robot_weld.base_rigid_id,j1_rough_axis_direction,j2_rough_axis_direction)

    # start calibration
    # find transformation matrix from the base marker rigid body (defined in motiv) to the actual robot base
    calib_obj.run_calib(config_dir+'MA2010_marker_config.yaml') # save calib config to file
