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
from fitting_3dcircle import fitting_3dcircle

class CalibRobotBase:
    def __init__(self,mocap_url,marker_ids,j1_rough_axis_direction,j2_rough_axis_direction):

        self.mocap_url = mocap_url
        self.mocap_cli = RRN.ConnectService(self.mocap_url)
        
        self.marker_ids = marker_ids
        self.clear_samples()
        self.sample_threshold = 0.5 # mm

        self.j1_rough_axis_direction=j1_rough_axis_direction
        self.j2_rough_axis_direction=j2_rough_axis_direction

        self.collect_thread_end = True
    
    def clear_samples(self):

        self.marker_position_table = {}
        for i in range(len(self.marker_ids)):
            self.marker_position_table[self.marker_ids[i]]=[]

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
                if this_marker_id in self.marker_ids:
                    this_position = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['position']))
                    if len(self.marker_position_table[this_marker_id]) == 0:
                        self.marker_position_table[this_marker_id].append(this_position)
                    else:
                        last_position = np.array(list(self.marker_position_table[this_marker_id][-1]))
                        if np.linalg.norm(this_position-last_position)>=self.sample_threshold:
                            self.marker_position_table[this_marker_id].append(this_position)
        sensor_data_srv.Close()
        # print("end thread")
    def detect_axis(self,rough_axis_direction):

        all_normals=[]
        all_centers=[]
        for i in range(len(self.marker_ids)):
            center, normal = fitting_3dcircle(self.marker_position_table[self.marker_ids[i]])
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
        print(z_axis)
        print(x_axis)
        x_axis = x_axis-np.dot(x_axis,z_axis)*z_axis
        print(z_axis)
        print(x_axis)
        x_axis = x_axis/np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis,x_axis)
        y_axis = y_axis/np.linalg.norm(y_axis)

        T_base = Transform(np.array([x_axis,y_axis,z_axis]).T,j1_center)

        return T_base

    def run_calib(self):

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
        print("total sample",len(self.marker_position_table[self.marker_ids[0]]))
        j1_p,j1_normal = self.detect_axis(self.j1_rough_axis_direction)

        self.clear_samples()
        if len(self.marker_position_table[self.marker_ids[0]]) != 0:
            raise("problem!!!")
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
        print("total sample",len(self.marker_position_table[self.marker_ids[0]]))
        j2_p,j2_normal = self.detect_axis(self.j2_rough_axis_direction)

        T_base = self.find_base(j1_p,j1_normal,j2_p,j2_normal)
        T_base_H = H_from_RT(T_base.R,T_base.p)

        np.savetxt('../config/MA2010_mocap_pose.csv',T_base_H,delimiter=',')

if __name__=='__main__':

    marker_ids = ['marker1_rigid1','marker2_rigid1','marker3_rigid1']
    mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'

    j1_rough_axis_direction = np.array([0,1,0])
    j2_rough_axis_direction = np.array([1,0,0])
    calib_obj = CalibRobotBase(mocap_url,marker_ids,j1_rough_axis_direction,j2_rough_axis_direction)

    # start calibration
    calib_obj.run_calib()