import numpy as np
from general_robotics_toolbox import *
from RobotRaconteur.Client import *
import time
from threading import Thread

class MocapPoseListener():
    def __init__(self,rr_mocap,robots):

        self.rr_mocap = rr_mocap

        self.robots = {}
        for i in range(len(robots)):
            self.robots[robots.robot_name] = robots[i]
        self.robots_base = {}
        for i in range(len(robots)):
            self.robots_base[robots.robot_name] = None
        self.clear_traj()

        self.collect_thread_end = True

    def clear_traj(self):
        self.robots_traj_p={}
        self.robots_traj_R={}
        self.robots_traj_stamps={}
        for robot_name in range(len(self.robots.keys())):
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
            for i in range(len(data.fiducials.recognized_fiducials)):
                this_marker_id = data.fiducials.recognized_fiducials[i].fiducial_marker
                for robot_name in range(len(self.robots.keys())):
                    if this_marker_id == self.robots[robot_name].base_markers_id.split('_')[0]:
                        this_position = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['position']))
                        this_orientation = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['orientation']))
                        # (T^basemarker_world * T^base_basemarker).inv()
                        # T^world_base
                        self.robots_base[robot_name] = Transform(q2R(this_orientation),this_position)*self.robots[robot_name].T_base_basemarker
                        self.robots_base[robot_name] = self.robots_base[robot_name].inv()
                    if this_marker_id == self.robots[robot_name].tool_markers_id.split('_')[0]:
                        if self.robots_base[robot_name] is not None:
                            this_position = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['position']))
                            this_orientation = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['orientation']))
                            # T^toolmarker_world * T^tool_toolmarker
                            # T^tool_world
                            T_tool_world = Transform(q2R(this_orientation),this_position)*self.robots[robot_name].T_tool_toolmarker
                            # T^world_base*T^tool_world
                            T_tool_base = self.robots_base[robot_name]*T_tool_world
                            self.robots_traj_p[robot_name].append(T_tool_base.p)
                            self.robots_traj_R[robot_name].append(T_tool_base.R)
                            self.robots_traj_stamps[robot_name].append(float(data.sensor_data.ts[0]['seconds'])+data.sensor_data.ts[0]['nanoseconds']*1e-9)

    def run_pose_listener(self):

        # clear previous logged data
        self.clear_traj()
        # start a new collect point/pose thread
        self.cp_thread = Thread( target = self.collect_point_thread)
        self.collect_thread_end = False
        self.cp_thread.start()

    def stop_pose_listener(self):
        
        # end the thread
        self.collect_thread_end = True
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

    mpl_obj = MocapPoseListener(mocap_cli,[robot_weld])
    mpl_obj.run_pose_listener()
    time.sleep(5)
    mpl_obj.stop_pose_listener()
    curve_p,curve_R,timestamps = mpl_obj.get_robots_traj()

    ## robots traj
    print('curve p:',curve_p[robot_weld.robot_name][:10])
    print('curve R:',curve_R[robot_weld.robot_name][:10])
    print('curve stamps:',timestamps[robot_weld.robot_name][:10])
