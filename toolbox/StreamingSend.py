import numpy as np
import time

class StreamingSend(object):
	def __init__(self,RR_robot,RR_robot_state,RobotJointCommand,streaming_rate=125.):
		self.RR_robot=RR_robot
		self.RR_robot_state=RR_robot_state
		self.RobotJointCommand=RobotJointCommand
		self.streaming_rate=streaming_rate
		self.command_seqno=0

	def get_breapoints(self,lam,vd):
		lam_diff=np.diff(lam)
		displacement=vd/self.streaming_rate
		breakpoints=[0]
		for i in range(1,len(lam)):
			if np.sum(lam_diff[breakpoints[-1]:i])>displacement:
				breakpoints.append(i)
		
		return np.array(breakpoints)

	def position_cmd(self,qd,start_time):
		###streaming points at every 8ms
		res, robot_state, _ = self.RR_robot_state.TryGetInValue()

		# Increment command_seqno
		self.command_seqno += 1

		# Create Fill the RobotJointCommand structure
		joint_cmd1 = self.RobotJointCommand()
		joint_cmd1.seqno = self.command_seqno # Strictly increasing command_seqno
		joint_cmd1.state_seqno = robot_state.seqno # Send current robot_state.seqno as failsafe
		
		
		# Set the joint command
		joint_cmd1.command = qd

		# Send the joint command to the robot
		self.RR_robot.position_command.PokeOutValue(joint_cmd1)

		# rate.Sleep()
		# time.sleep(0.008)
		while time.time()-start_time<1/self.streaming_rate-0.0002:
			continue
		
		return float(robot_state.ts['microseconds'])/1e6, robot_state.joint_position


	def jog2q(self,qd):
		###JOG TO starting pose first
		res, robot_state, _ = self.RR_robot_state.TryGetInValue()
		q_cur=robot_state.joint_position
		num_points_jogging=self.streaming_rate*np.max(np.abs(q_cur-qd))/0.2


		for j in range(int(num_points_jogging)):
			q_target = (q_cur*(num_points_jogging-j))/num_points_jogging+qd*j/num_points_jogging
			self.position_cmd(q_target,time.time())
			
		###init point wait
		for i in range(20):
			self.position_cmd(qd,time.time())


	def traj_streaming(self,curve_js):
		joint_recording=[]
		timestamp_recording=[]
		for i in range(len(curve_js)):
			ts,js=self.position_cmd(curve_js[i],time.time())
			timestamp_recording.append(ts)
			joint_recording.append(js)
		
		return np.array(timestamp_recording), np.array(joint_recording)

	def traj_tracking(self,lam,curve_js,vd,ctrl_joints):
		###lam: curve length parameter
		###curve_js: joint space trajectory
		###vd: desired velocity
		###ctrl_joints: joints to be controlled, array of 0 and 1

		joint_recording=[]
		timestamp_recording=[]
		res, robot_state, _ = self.RR_robot_state.TryGetInValue()
		q_cur=np.take(robot_state.joint_position,ctrl_joints.nonzero()[0])
		q_static=np.take(robot_state.joint_position,(~ctrl_joints.astype(bool)).astype(int).nonzero()[0])
		traj_start_time=time.time()
		while True:
			now=time.time()

			lam_now=(now-traj_start_time)*vd
			lam_idx=np.searchsorted(lam,lam_now)-1
			lam_next=lam_now+vd/self.streaming_rate
			lam_next_idx=np.searchsorted(lam,lam_next)-1
			if lam_next>lam[-1]:
				break 		###end of trajectory
			###interpolate desired joint position
			qd_cur=curve_js[lam_idx]*(lam_now-lam[lam_idx])/(lam[lam_idx+1]-lam[lam_idx])+curve_js[lam_idx+1]*(lam[lam_idx+1]-lam_now)/(lam[lam_idx+1]-lam[lam_idx])
			###interpolate next desired joint position
			qd_next=curve_js[lam_next_idx]*(lam_next-lam[lam_next_idx])/(lam[lam_next_idx+1]-lam[lam_next_idx])+curve_js[lam_next_idx+1]*(lam[lam_next_idx+1]-lam_next)/(lam[lam_next_idx+1]-lam[lam_next_idx])

			curve_js_cmd=np.ones(len(robot_state.joint_position))
			np.put(curve_js_cmd,(~ctrl_joints.astype(bool)).astype(int).nonzero()[0],q_static)
			if len(joint_recording)==0:
				np.put(curve_js_cmd,ctrl_joints.nonzero()[0],qd_next)
			else:
				np.put(curve_js_cmd,ctrl_joints.nonzero()[0],qd_next+(qd_cur-q_cur)*0.1)

			# print(curve_js_cmd)
			ts,js=self.position_cmd(curve_js_cmd,now)
			timestamp_recording.append(ts)
			joint_recording.append(js)
			q_cur=np.take(js,ctrl_joints.nonzero()[0])

		return np.array(timestamp_recording), np.array(joint_recording)