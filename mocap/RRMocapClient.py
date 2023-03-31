from RobotRaconteur.Client import *

class RRMocapClient():
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

        pass
