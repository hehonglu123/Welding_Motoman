import numpy as np
from general_robotics_toolbox import *
from RobotRaconteur.Client import *
import time
from threading import Thread
import sys
sys.path.append('../toolbox/')
sys.path.append('../redundancy_resolution/')
from utils import *
from robot_def import * 

class MocapPoseListener():
    def __init__(self,rr_mocap,robots,collect_base_stop=1e4):

        self.rr_mocap = rr_mocap

        self.robots = {}
        for i in range(len(robots)):
            self.robots[robots[i].robot_name] = robots[i]
        self.robots_base = {}
        self.robots_base_p = {}
        self.robots_base_rpy = {}
        self.robots_base_rpy_bias = {}
        for i in range(len(robots)):
            self.robots_base[robots[i].robot_name] = None
            self.robots_base_p[robots[i].robot_name] = []
            self.robots_base_rpy[robots[i].robot_name] = []
            self.robots_base_rpy_bias[robots[i].robot_name] = [0,0,0]
        self.clear_traj()

        # threading
        self.cp_thread = None
        self.collect_thread_end = True
        self.collect_marker = False

        # determined base pose
        self.collect_base_stop = collect_base_stop

    def clear_traj(self):
        self.robots_traj_p={}
        self.robots_traj_R={}
        self.robots_traj_stamps={}
        for robot_name in self.robots.keys():
            self.robots_traj_p[robot_name] = []
            self.robots_traj_R[robot_name] = []
            self.robots_traj_stamps[robot_name]=[]
    
    def collect_point_thread(self):

        sensor_data_srv = self.rr_mocap.fiducials_sensor_data.Connect(-1)
        while sensor_data_srv.Available>=1:
            sensor_data_srv.ReceivePacket()
        while not self.collect_thread_end:
            try:
                data = sensor_data_srv.ReceivePacketWait(timeout=10)
            except:
                continue
            if self.collect_marker:
                for i in range(len(data.fiducials.recognized_fiducials)):
                    this_marker_id = data.fiducials.recognized_fiducials[i].fiducial_marker
                    for robot_name in self.robots.keys():
                        if this_marker_id == self.robots[robot_name].base_rigid_id and len(self.robots_base_p[robot_name])<self.collect_base_stop:
                            this_position = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['position']))
                            if np.all(this_position == np.array([0.,0.,0.])):
                                continue
                            this_orientation_rpy = R2rpy(q2R(np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['orientation']))))
                            self.robots_base_p[robot_name].append(this_position)
                            self.robots_base_rpy[robot_name].append(this_orientation_rpy)

                            # (T^basemarker_world * T^base_basemarker).inv()
                            # T^world_base
                            pos_ave = np.mean(self.robots_base_p[robot_name],axis=0)
                            rpy_ave = np.mean(self.robots_base_rpy[robot_name],axis=0)
                            self.robots_base[robot_name] = Transform(rpy2R(rpy_ave),pos_ave)*self.robots[robot_name].T_base_basemarker
                            self.robots_base[robot_name] = self.robots_base[robot_name].inv()
                        if this_marker_id == self.robots[robot_name].tool_rigid_id:
                            if self.robots_base[robot_name] is not None:
                                this_position = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['position']))
                                if np.all(this_position == np.array([0.,0.,0.])):
                                    continue
                                this_orientation = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['orientation']))
                                # T^toolmarker_world * T^tool_toolmarker
                                # T^tool_world
                                T_tool_world = Transform(q2R(this_orientation),this_position)*self.robots[robot_name].T_tool_toolmarker
                                # T^world_base*T^tool_world
                                T_tool_base = self.robots_base[robot_name]*T_tool_world
                                self.robots_traj_p[robot_name].append(T_tool_base.p)
                                self.robots_traj_R[robot_name].append(T_tool_base.R)
                                self.robots_traj_stamps[robot_name].append(float(data.sensor_data.ts[0]['seconds'])+data.sensor_data.ts[0]['nanoseconds']*1e-9)
        sensor_data_srv.Close()

    def run_pose_listener(self):

        # clear previous logged data
        self.clear_traj()
        if self.cp_thread is None:
            # start a new collect point/pose thread
            self.cp_thread = Thread( target = self.collect_point_thread,daemon=True)
            self.collect_thread_end = False
            self.cp_thread.start()
        self.collect_marker=True

    def stop_pose_listener(self):
        
        self.collect_marker = False
        
    def end_pose_listener(self):

        # end the thread
        self.collect_marker=False
        self.collect_thread_end = True
        if self.cp_thread is not None:
            self.cp_thread.join()

    def get_robots_traj(self):

        return self.robots_traj_p,self.robots_traj_R,self.robots_traj_stamps

if __name__=='__main__':

    config_dir='../config/'
    robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg.csv',\
    base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
    mocap_url = mocap_url
    mocap_cli = RRN.ConnectService(mocap_url)

    mpl_obj = MocapPoseListener(mocap_cli,[robot_weld],collect_base_stop=1000)
    mpl_obj.run_pose_listener()
    time.sleep(5)
    mpl_obj.stop_pose_listener()
    curve_p,curve_R,timestamps = mpl_obj.get_robots_traj()

    ## robots traj
    print(curve_p[robot_weld.robot_name][0][1])
    print('curve p:',curve_p[robot_weld.robot_name][:10])
    print('curve R:',curve_R[robot_weld.robot_name][:10])
    print('curve stamps:',timestamps[robot_weld.robot_name][:10])

    #### run second time
    mpl_obj.run_pose_listener()
    time.sleep(5)
    mpl_obj.stop_pose_listener()
    curve_p,curve_R,timestamps = mpl_obj.get_robots_traj()
    ## robots traj
    print('curve p:',curve_p[robot_weld.robot_name][:10])
    print('curve R:',curve_R[robot_weld.robot_name][:10])
    print('curve stamps:',timestamps[robot_weld.robot_name][:10])
    print(curve_p[robot_weld.robot_name][-1][1])

    mpl_obj.end_pose_listener()
    
