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
    def __init__(self,rr_mocap,robots,collect_base_window=240,use_static_base=False,use_toolmarker_rigid=False):

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
        self.collect_base_window = collect_base_window
        self.use_static_base = use_static_base
        self.use_toolmarker_rigid = use_toolmarker_rigid
        if use_static_base:
            for robot_name in self.robots_base:
                self.robots_base[robot_name] = self.robots[robot_name].T_base_mocap.inv()

    def clear_traj(self):
        self.robots_traj_p={}
        self.robots_traj_R={}
        self.robots_tool_mocap_p={}
        self.robots_tool_mocap_R={}
        self.robots_base_mocap_p={}
        self.robots_base_mocap_R={}
        self.robots_traj_stamps={}
        for robot_name in self.robots.keys():
            self.robots_traj_p[robot_name] = []
            self.robots_traj_R[robot_name] = []
            self.robots_traj_stamps[robot_name]=[]
            self.robots_tool_mocap_p[robot_name]=[]
            self.robots_tool_mocap_R[robot_name]=[]
            self.robots_base_mocap_p[robot_name]=[]
            self.robots_base_mocap_R[robot_name]=[]
    
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
                        if this_marker_id == self.robots[robot_name].base_rigid_id and not self.use_static_base:
                            this_position = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['position']))
                            if np.all(this_position == np.array([0.,0.,0.])):
                                continue
                            this_orientation_R = q2R(np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['orientation'])))
                            this_orientation_rpy = R2rpy(this_orientation_R)
                            self.robots_base_p[robot_name].append(this_position)
                            self.robots_base_rpy[robot_name].append(this_orientation_rpy)

                            if len(self.robots_base_p[robot_name])>self.collect_base_window:
                                self.robots_base_p[robot_name].pop(0)
                                self.robots_base_rpy[robot_name].pop(0)

                            # (T^basemarker_world * T^base_basemarker).inv()
                            # T^world_base
                            pos_ave = np.mean(self.robots_base_p[robot_name],axis=0)
                            rpy_ave = np.mean(self.robots_base_rpy[robot_name],axis=0)
                            self.robots_base[robot_name] = Transform(rpy2R(rpy_ave),pos_ave)*self.robots[robot_name].T_base_basemarker
                            self.robots_base[robot_name] = self.robots_base[robot_name].inv()

                            self.robots_base_mocap_p[robot_name].append(this_position)
                            self.robots_base_mocap_R[robot_name].append(this_orientation_R)
                        if this_marker_id == self.robots[robot_name].tool_rigid_id:
                            if self.robots_base[robot_name] is not None:
                                this_position = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['position']))
                                if np.all(this_position == np.array([0.,0.,0.])):
                                    continue
                                this_orientation = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['orientation']))
                                # T^toolmarker_world * T^tool_toolmarker
                                # T^tool_world
                                if not self.use_toolmarker_rigid:
                                    T_tool_world = Transform(q2R(this_orientation),this_position)*self.robots[robot_name].T_tool_toolmarker
                                else:
                                    T_tool_world = Transform(q2R(this_orientation),this_position)
                                # T^world_base*T^tool_world
                                T_tool_base = self.robots_base[robot_name]*T_tool_world
                                self.robots_traj_p[robot_name].append(T_tool_base.p)
                                self.robots_traj_R[robot_name].append(T_tool_base.R)
                                self.robots_traj_stamps[robot_name].append(float(data.sensor_data.ts[0]['seconds'])+data.sensor_data.ts[0]['nanoseconds']*1e-9)

                                self.robots_tool_mocap_p[robot_name].append(this_position)
                                self.robots_tool_mocap_R[robot_name].append(q2R(this_orientation))

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
    
    def get_rigid_pose(self):

        return self.robots_base_mocap_p,self.robots_base_mocap_R,self.robots_tool_mocap_p,self.robots_tool_mocap_R,self.robots_traj_stamps

class MocapFrameListener():
    def __init__(self,rr_mocap,target_frames,source_frame='world'):

        self.rr_mocap = rr_mocap
        
        self.source_frame = source_frame
        self.target_frames = target_frames
        self.clear_traj()

        # threading
        self.cp_thread = None
        self.collect_thread_end = True
        self.collect_marker = False

    def clear_traj(self):
        self.target_frames_traj_p={}
        self.target_frames_traj_R={}
        self.traj_stamps={}
        for frame_name in self.target_frames:
            self.target_frames_traj_p[frame_name] = []
            self.target_frames_traj_R[frame_name] = []
            self.traj_stamps[frame_name]=[]
    
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
                T_world_source = None
                if self.source_frame=='world':
                    T_world_source = Transform(np.eye(3),[0,0,0])
                else:
                    for i in range(len(data.fiducials.recognized_fiducials)):
                        this_marker_id = data.fiducials.recognized_fiducials[i].fiducial_marker
                        if this_marker_id == self.source_frame:
                            this_position = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['position']))
                            if np.all(this_position == np.array([0.,0.,0.])):
                                continue
                            this_orientation_R = q2R(np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['orientation'])))
                            T_world_source = Transform(this_orientation_R,this_position).inv()
                            break
                if T_world_source is None:
                    continue
                for i in range(len(data.fiducials.recognized_fiducials)):
                    this_marker_id = data.fiducials.recognized_fiducials[i].fiducial_marker
                    if this_marker_id in self.target_frames:
                        this_position = np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['position']))
                        if np.all(this_position == np.array([0.,0.,0.])):
                            continue
                        this_orientation_R = q2R(np.array(list(data.fiducials.recognized_fiducials[i].pose.pose.pose[0]['orientation'])))
                        T_target_world = Transform(this_orientation_R,this_position)
                        T_target_source = T_world_source*T_target_world
                        self.target_frames_traj_p[this_marker_id].append(T_target_source.p)
                        self.target_frames_traj_R[this_marker_id].append(T_target_source.R)
                        self.traj_stamps[this_marker_id].append(float(data.sensor_data.ts[0]['seconds'])+data.sensor_data.ts[0]['nanoseconds']*1e-9)

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

    def get_frames_traj(self):

        return self.target_frames_traj_p,self.target_frames_traj_R,self.traj_stamps

def robotposelistener():
    config_dir='../config/'
    robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg.csv',\
    base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

    mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
    mocap_url = mocap_url
    mocap_cli = RRN.ConnectService(mocap_url)

    # collect base window windown length of moving average
    mpl_obj = MocapPoseListener(mocap_cli,[robot_weld],collect_base_window=240)
    # mpl_obj = MocapPoseListener(mocap_cli,[robot_weld],collect_base_window=1,use_static_base=True)
    mpl_obj.run_pose_listener()
    time.sleep(1)
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

def framelistener():
    
    mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
    mocap_url = mocap_url
    mocap_cli = RRN.ConnectService(mocap_url)

    mpl_obj = MocapFrameListener(mocap_cli,['rigid3'],'rigid7')
    mpl_obj.run_pose_listener()
    time.sleep(5)
    mpl_obj.stop_pose_listener()

    curve_p,curve_R,timestamps = mpl_obj.get_frames_traj()
    ## robots traj
    print('curve p:',np.mean(curve_p['rigid3'],axis=0))
    print('std curve p:',np.std(curve_p['rigid3'],axis=0))
    print('curve R:',curve_R['rigid3'][-1])
    print('curve stamps:',timestamps['rigid3'][-1])

def markerlistener():
    
    mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
    mocap_url = mocap_url
    mocap_cli = RRN.ConnectService(mocap_url)

    mpl_obj = MocapFrameListener(mocap_cli,['marker1_rigid3'],'world')
    mpl_obj.run_pose_listener()
    time.sleep(5)
    mpl_obj.stop_pose_listener()

    curve_p,curve_R,timestamps = mpl_obj.get_frames_traj()
    ## robots traj
    print('curve p:',np.mean(curve_p['marker1_rigid3'],axis=0))
    print('std curve p:',np.std(curve_p['marker1_rigid3'],axis=0))
    print('curve R:',curve_R['marker1_rigid3'][-1])
    print('curve stamps:',timestamps['marker1_rigid3'][:5])

if __name__=='__main__':

    # robotposelistener()
    # framelistener()
    markerlistener()

