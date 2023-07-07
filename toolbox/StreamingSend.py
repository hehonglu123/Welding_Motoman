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

	def position_cmd(self,qd):
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
		now=time.time()
		while time.time()-now<1/self.streaming_rate-0.0002:
			continue
		
		return float(robot_state.ts['microseconds'])/1e6, robot_state.joint_position


	def jog2q(self,qd):

		###JOG TO starting pose first
		res, robot_state, _ = self.RR_robot_state.TryGetInValue()
		q_cur=robot_state.joint_position
		num_points_jogging=np.linalg.norm(q_cur-qd)/0.002


		for j in range(int(num_points_jogging)):
			q_target = (q_cur*(num_points_jogging-j))/num_points_jogging+qd*j/num_points_jogging
			self.position_cmd(q_target)
			
		###init point wait
		for i in range(20):
			self.position_cmd(qd)


	def traj_streaming(self,curve_js):
		joint_recording=[]
		timestamp_recording=[]
		for i in range(len(curve_js)):
			ts,js=self.position_cmd(curve_js[i])
			timestamp_recording.append(ts)
			joint_recording.append(js)
		
		return np.array(timestamp_recording), np.array(joint_recording)

