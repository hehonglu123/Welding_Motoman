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

class CalibRobotBase:
    def __init__(self,mocap_cli,calib_marker_ids,base_marker_ids,base_rigid_id,j1_rough_axis_direction,j2_rough_axis_direction,positioner=False):

        self.mocap_cli = mocap_cli
        
        self.calib_marker_ids = calib_marker_ids
        self.base_markers_ids = base_marker_ids
        self.base_rigid_id = base_rigid_id
        self.clear_samples()
        self.sample_threshold = 1e-10 # mm

        self.j1_rough_axis_direction=j1_rough_axis_direction
        self.j2_rough_axis_direction=j2_rough_axis_direction
        self.positioner = positioner

        self.collect_markers = False
        self.collect_thread_end = True
    
    def clear_samples(self):

        self.marker_position_table = {}
        self.marker_orientation_table = {}
        for marker_id in self.calib_marker_ids:
            self.marker_position_table[marker_id]=[]
            self.marker_orientation_table[marker_id]=[]
        for marker_id in self.base_markers_ids:
            self.marker_position_table[marker_id]=[]
            self.marker_orientation_table[marker_id]=[]
        self.marker_position_table[self.base_rigid_id]=[]
        self.marker_orientation_table[self.base_rigid_id]=[]

    def collect_point_thread(self):
        
        sensor_data_srv = self.mocap_cli.fiducials_sensor_data.Connect(-1)
        while sensor_data_srv.Available>=1:
            sensor_data_srv.ReceivePacket()
        while not self.collect_thread_end:
            try:
                data = sensor_data_srv.ReceivePacketWait(timeout=10)
            except:
                continue
            if self.collect_markers:
                for i in range(len(data.fiducials.recognized_fiducials)):
                    this_marker_id = data.fiducials.recognized_fiducials[i].fiducial_marker
                    if this_marker_id in self.calib_marker_ids or this_marker_id in self.base_markers_ids or this_marker_id==self.base_rigid_id:
                        this_position = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['position']))
                        this_orientation = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['orientation']))
                        if np.all(this_position == np.array([0.,0.,0.])):
                            continue
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

        if not self.positioner: # robot axis definition
            z_axis = j1_normal
            y_axis = j2_normal
            y_axis = y_axis-np.dot(y_axis,z_axis)*z_axis
            y_axis = y_axis/np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis,z_axis)
            x_axis = x_axis/np.linalg.norm(x_axis)

            # x_axis = j2_center-j1_center
            # x_axis = x_axis-np.dot(x_axis,z_axis)*z_axis
            # x_axis = x_axis/np.linalg.norm(x_axis)
            # y_axis = np.cross(z_axis,x_axis)
            # y_axis = y_axis/np.linalg.norm(y_axis)
        else: # positioner axis definition
            z_axis = j2_normal
            y_axis = j1_normal
            y_axis = y_axis-np.dot(y_axis,z_axis)*z_axis
            y_axis = y_axis/np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis,z_axis)
            x_axis = x_axis/np.linalg.norm(x_axis)

        T_base_mocap = Transform(np.array([x_axis,y_axis,z_axis]).T,j1_center)

        base_orientation_rpy = []
        for quat in self.marker_orientation_table[self.base_rigid_id]:
            base_orientation_rpy.append(R2rpy(q2R(quat)))

        T_basemarker_mocap = Transform(rpy2R(np.mean(base_orientation_rpy,axis=0)),np.mean(self.marker_position_table[self.base_rigid_id],axis=0))

        T_base_basemarker = T_basemarker_mocap.inv()*T_base_mocap

        return T_base_mocap,T_base_basemarker

    def run_calib(self,base_marker_config_file,auto=False,rob_IP=None,ROBOT_CHOICE=None,rob_p2d=None,paths=[],rob_speed=3,repeat_N=1):

        client=MotionProgramExecClient()

        # check where's joint axis 1
        self.clear_samples()
        cp_thread = Thread( target = self.collect_point_thread,daemon=True)
        self.collect_thread_end = False
        cp_thread.start()
        input("Press Enter and start moving ONLY J1")
        time.sleep(0.5)
        if auto:
            print("Robot is collecting samples")
            # move robot to start
            mp=MotionProgram(ROBOT_CHOICE=ROBOT_CHOICE,pulse2deg=rob_p2d)
            mp.MoveJ(paths[0][0],rob_speed,0)
            client.execute_motion_program(mp)
            # collect data
            self.collect_markers = True
            time.sleep(0.5)
            mp=MotionProgram(ROBOT_CHOICE=ROBOT_CHOICE,pulse2deg=rob_p2d)
            for N in range(repeat_N):
                mp.MoveJ(paths[0][1],rob_speed,0)
                mp.MoveJ(paths[0][0],rob_speed,0)
            client.execute_motion_program(mp)
        else:
            self.collect_markers = True
            input("Press Enter if you collect enough samples")
        time.sleep(0.5)
        self.collect_markers = False
        for i in range(len(self.calib_marker_ids)):
            if len(self.marker_position_table[self.calib_marker_ids[i]]) == 0:
                raise("problem!!! No J1 Calib markers!")
        print("total sample",len(self.marker_position_table[self.calib_marker_ids[0]]))
        j1_p,j1_normal = self.detect_axis(self.j1_rough_axis_direction)

        self.clear_samples()
        if len(self.marker_position_table[self.calib_marker_ids[0]]) != 0:
            raise("problem!!! Did not clear J1 Calib markers")
        # check where's joint axis 2
        input("Press Enter and start moving ONLY J2")
        time.sleep(0.5)
        if auto:
            print("Robot is collecting samples")
            # move robot to start
            mp=MotionProgram(ROBOT_CHOICE=ROBOT_CHOICE,pulse2deg=rob_p2d)
            mp.MoveJ(paths[1][0],rob_speed,0)
            client.execute_motion_program(mp)
            # collect data
            self.collect_markers = True
            time.sleep(0.5)
            mp=MotionProgram(ROBOT_CHOICE=ROBOT_CHOICE,pulse2deg=rob_p2d)
            for N in range(repeat_N):
                mp.MoveJ(paths[1][1],rob_speed,0)
                mp.MoveJ(paths[1][0],rob_speed,0)
            client.execute_motion_program(mp)
        else:
            self.collect_markers = True
            input("Press Enter if you collect enough samples")
        time.sleep(0.5)
        self.collect_markers = False
        for i in range(len(self.calib_marker_ids)):
            if len(self.marker_position_table[self.calib_marker_ids[i]]) == 0:
                raise("problem!!! No J2 Calib markers!")
        print("total sample",len(self.marker_position_table[self.calib_marker_ids[0]]))
        j2_p,j2_normal = self.detect_axis(self.j2_rough_axis_direction)

        if len(self.marker_position_table[self.base_rigid_id]) == 0:
            self.collect_markers = True
            time.sleep(3)
            self.collect_markers = False
        if len(self.marker_position_table[self.base_rigid_id]) == 0:
            raise("problem!!! No base rigid!")

        T_base_mocap,T_base_basemarker = self.find_base(j1_p,j1_normal,j2_p,j2_normal) # T^base_mocap, T^base_basemarker
        T_base_H = H_from_RT(T_base_basemarker.R,T_base_basemarker.p)

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
        base_marker_data['calib_base_mocap_pose']['position']['x'] = float(T_base_mocap.p[0])
        base_marker_data['calib_base_mocap_pose']['position']['y'] = float(T_base_mocap.p[1])
        base_marker_data['calib_base_mocap_pose']['position']['z'] = float(T_base_mocap.p[2])
        quat = R2q(T_base_mocap.R)
        base_marker_data['calib_base_mocap_pose']['orientation'] = {}
        base_marker_data['calib_base_mocap_pose']['orientation']['w'] = float(quat[0])
        base_marker_data['calib_base_mocap_pose']['orientation']['x'] = float(quat[1])
        base_marker_data['calib_base_mocap_pose']['orientation']['y'] = float(quat[2])
        base_marker_data['calib_base_mocap_pose']['orientation']['z'] = float(quat[3])

        with open(base_marker_config_file,'w') as file:
            yaml.safe_dump(base_marker_data,file)
        
        self.collect_thread_end=True
        cp_thread.join()
        print("Result T_base_basemarker:")
        print(T_base_basemarker)
        print("Done. Please check file:",base_marker_config_file)

def calib_S1():

    auto = True

    config_dir='../config/'
    turn_table=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv'\
        ,pulse2deg_file_path=config_dir+'D500B_pulse2deg.csv',base_marker_config_file=config_dir+'D500B_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
    mocap_cli = RRN.ConnectService(mocap_url)

    j1_rough_axis_direction = np.array([0,0,1])
    j2_rough_axis_direction = np.array([0,1,0])
    calib_obj = CalibRobotBase(mocap_cli,turn_table.calib_markers_id,turn_table.base_markers_id,turn_table.base_rigid_id,j1_rough_axis_direction,j2_rough_axis_direction)

    q1_1=np.array([-15,180])
    q1_2=np.array([-15,180])
    q2_1=np.array([-15,180])
    q2_2=np.array([-15,180])
    q_paths = [[q1_1,q1_2],[q2_1,q2_2]]

    # start calibration
    # find transformation matrix from the base marker rigid body (defined in motiv) to the actual robot base
    calib_obj.run_calib(config_dir+'MA1440_marker_config.yaml',auto,'192.168.1.31','ST1',turn_table.pulse2deg,q_paths) # save calib config to file

def calib_R2():

    auto = True

    config_dir='../config/'
    robot=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'scanner_tcp2.csv',\
	pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg.csv',base_marker_config_file=config_dir+'MA1440_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
    mocap_cli = RRN.ConnectService(mocap_url)

    j1_rough_axis_direction = np.array([0,1,0])
    j2_rough_axis_direction = np.array([-1,0,0])
    calib_obj = CalibRobotBase(mocap_cli,robot.calib_markers_id,robot.base_markers_id,robot.base_rigid_id,j1_rough_axis_direction,j2_rough_axis_direction)

    q1_1=np.array([-21.2066,-23.0282,0,0,11,0])
    q1_2=np.array([114.9,-23.0282,0,0,11,0])
    q2_1=np.array([73,42,0,0,-48,0])
    q2_2=np.array([73,-81,0,0,-48,0])
    q_paths = [[q1_1,q1_2],[q2_1,q2_2]]

    # start calibration
    # find transformation matrix from the base marker rigid body (defined in motiv) to the actual robot base
    calib_obj.run_calib(config_dir+'MA1440_marker_config.yaml',auto,'192.168.1.31','RB2',robot.pulse2deg,q_paths) # save calib config to file

def calib_R1():

    auto = True

    config_dir='../config/'
    robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',base_marker_config_file=config_dir+'MA2010_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
    mocap_cli = RRN.ConnectService(mocap_url)

    j1_rough_axis_direction = np.array([0,1,0])
    j2_rough_axis_direction = np.array([1,0,0])
    calib_obj = CalibRobotBase(mocap_cli,robot_weld.calib_markers_id,robot_weld.base_markers_id,robot_weld.base_rigid_id,j1_rough_axis_direction,j2_rough_axis_direction)

    q1_1=np.array([-42.6985,-50.2617,-9.1801,0,-47.6771,0])
    q1_2=np.array([95.1139,-50.2617,-9.1801,0,-47.6771,0])
    q2_1=np.array([0.4003,-60.2277,-9.1801,0,-47.6771,0])
    q2_2=np.array([0.4003,22.1919,-9.1801,0,-47.6771,0])
    q_paths = [[q1_1,q1_2],[q2_1,q2_2]]

    # start calibration
    # find transformation matrix from the base marker rigid body (defined in motiv) to the actual robot base
    calib_obj.run_calib(config_dir+'MA2010_marker_config.yaml',auto,'192.168.1.31','RB1',robot_weld.pulse2deg,q_paths,rob_speed=3,repeat_N=1) # save calib config to file

if __name__=='__main__':

    calib_R1()
    # calib_R2()
