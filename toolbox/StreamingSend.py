import numpy as np
import time, copy

class StreamingSend(object):
	def __init__(self,RR_robot,RR_robot_state,RobotJointCommand,streaming_rate=125.,latency=0.1):
		self.RR_robot=RR_robot
		self.RR_robot_state=RR_robot_state
		self.RobotJointCommand=RobotJointCommand
		self.streaming_rate=streaming_rate
		self.command_seqno=0

	# def get_breakpoints(self,lam,vd):
	# 	###get breakpoints indices with dense lam and vd
	# 	lam_diff=np.diff(lam)
	# 	displacement=vd/self.streaming_rate
	# 	breakpoints=[0]
	# 	for i in range(1,len(lam)):
	# 		if np.sum(lam_diff[breakpoints[-1]:i])>displacement:
	# 			breakpoints.append(i)
		
	# 	return np.array(breakpoints)

	def get_breakpoints(self, lam, vd):
		lam_diff = np.diff(lam)
		displacement = vd / self.streaming_rate
		cumulative_lam_diff = np.cumsum(lam_diff)

		# The mask gives True wherever the cumulative sum exceeds a multiple of displacement
		mask = np.diff(np.floor(cumulative_lam_diff / displacement))

		# Getting the indices where mask is True
		breakpoints = np.insert(np.where(mask)[0] + 1, 0, 0)

		return breakpoints

	def position_cmd(self,qd,start_time=None):
		###qd: joint position command
		###start_time: loop start time to make sure 8ms streaming rate, if None, then no wait
		robot_state = self.RR_robot_state.InValue

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

		if start_time:
			while time.time()-start_time<1/self.streaming_rate-0.0007:
				continue
		
		return float(robot_state.ts['microseconds'])/1e6, robot_state.joint_position


	def jog2q(self,qd,point_distance=0.2):
		###JOG TO starting pose first
		res, robot_state, _ = self.RR_robot_state.TryGetInValue()
		q_cur=robot_state.joint_position
		num_points_jogging=self.streaming_rate*np.max(np.abs(q_cur-qd))/point_distance


		for j in range(int(num_points_jogging)):
			q_target = (q_cur*(num_points_jogging-j))/num_points_jogging+qd*j/num_points_jogging
			self.position_cmd(q_target,time.time())
			
		###init point wait
		for i in range(20):
			self.position_cmd(qd,time.time())


	def traj_streaming(self,curve_js,ctrl_joints):
		###curve_js: Nxn, 2d joint space trajectory
		###ctrl_joints: joints to be controlled, array of 0 and 1

		joint_recording=[]
		timestamp_recording=[]
		res, robot_state, _ = self.RR_robot_state.TryGetInValue()
		q_static=np.take(robot_state.joint_position,(~ctrl_joints.astype(bool)).astype(int).nonzero()[0])
		for i in range(len(curve_js)):
			now=time.time()
			curve_js_cmd=np.zeros(len(robot_state.joint_position))
			np.put(curve_js_cmd,(~ctrl_joints.astype(bool)).astype(int).nonzero()[0],q_static)
			curve_js_cmd[ctrl_joints.nonzero()[0]]=curve_js[i]
			ts,js=self.position_cmd(curve_js_cmd,time.time())
			timestamp_recording.append(ts)
			try:
				joint_recording.append(js[ctrl_joints.nonzero()[0]])
			except:
				print(js,ts)
		#######################Wait for the robot to reach the last point with joint FEEDBACK#########################
		q_prev=joint_recording[-1]
		ts_prev=timestamp_recording[-1]
		counts=0
		while True:
			ts=float(self.RR_robot_state.InValue.ts['microseconds'])/1e6
			js=self.RR_robot_state.InValue.joint_position[ctrl_joints.nonzero()[0]]
			#only updates when the timestamp changes
			if ts_prev!=ts:
				if np.linalg.norm(js-q_prev)<0.0001:#if not moving
					counts+=1
				else:
					counts=0
				ts_prev=copy.deepcopy(ts)
				qs_prev=copy.deepcopy(js)
				joint_recording.append(js)
				timestamp_recording.append(ts)
				if counts>8:    ###in case getting static stale data 
					break
			q_prev=copy.deepcopy(js)

		timestamp_recording=np.array(timestamp_recording)
		timestamp_recording-=timestamp_recording[0]
		return timestamp_recording, np.array(joint_recording)

	# def traj_tracking_js(self,curve_js,ctrl_joints):
	# 	###joint space trajectory tracking with exact number of points at streaming_rate
	# 	###curve_js: Nxn, 2d joint space trajectory
	# 	###ctrl_joints: joints to be controlled, array of 0 and 1

	# 	joint_recording=[]
	# 	timestamp_recording=[]
	# 	res, robot_state, _ = self.RR_robot_state.TryGetInValue()
	# 	q_cur=np.take(robot_state.joint_position,ctrl_joints.nonzero()[0])
	# 	q_static=np.take(robot_state.joint_position,(~ctrl_joints.astype(bool)).astype(int).nonzero()[0])
	# 	for i in range(len(curve_js)):
	# 		now=time.time()
	# 		curve_js_cmd=np.zeros(len(robot_state.joint_position))
	# 		np.put(curve_js_cmd,(~ctrl_joints.astype(bool)).astype(int).nonzero()[0],q_static)
	# 		if len(joint_recording)==0:
	# 			curve_js_cmd[ctrl_joints.nonzero()[0]]=curve_js[i]
	# 		else:
	# 			curve_js_cmd[ctrl_joints.nonzero()[0]]=curve_js[i]+(curve_js[i]-q_cur)*0.1

	# 		ts,js=self.position_cmd(curve_js_cmd,now)
	# 		timestamp_recording.append(ts)
	# 		joint_recording.append(js[ctrl_joints.nonzero()[0]])
	# 		q_cur=js[ctrl_joints.nonzero()[0]]

	# 	timestamp_recording=np.array(timestamp_recording)
	# 	timestamp_recording-=timestamp_recording[0]
	# 	return timestamp_recording, np.array(joint_recording)

	# def traj_tracking_lam(self,lam,curve_js,vd,ctrl_joints):
	# 	####trajectory tracking with vd and lam parameters, curve_js may or may not be exact num_points at streaming_rate
	# 	###lam: curve length parameter
	# 	###curve_js: joint space trajectory
	# 	###vd: desired velocity
	# 	###ctrl_joints: joints to be controlled, array of 0 and 1

	# 	joint_recording=[]
	# 	timestamp_recording=[]
	# 	res, robot_state, _ = self.RR_robot_state.TryGetInValue()
	# 	q_cur=np.take(robot_state.joint_position,ctrl_joints.nonzero()[0])
	# 	q_static=np.take(robot_state.joint_position,(~ctrl_joints.astype(bool)).astype(int).nonzero()[0])
	# 	traj_start_time=time.time()
	# 	while True:
	# 		now=time.time()

	# 		lam_now=(now-traj_start_time)*vd
	# 		lam_idx=np.searchsorted(lam,lam_now)-1
	# 		lam_next=lam_now+vd/self.streaming_rate
	# 		lam_next_idx=np.searchsorted(lam,lam_next)-1
	# 		if lam_next>lam[-1]:
	# 			break 		###end of trajectory
	# 		###interpolate desired joint position
	# 		qd_cur=curve_js[lam_idx]*(lam_now-lam[lam_idx])/(lam[lam_idx+1]-lam[lam_idx])+curve_js[lam_idx+1]*(lam[lam_idx+1]-lam_now)/(lam[lam_idx+1]-lam[lam_idx])
	# 		###interpolate next desired joint position
	# 		qd_next=curve_js[lam_next_idx]*(lam_next-lam[lam_next_idx])/(lam[lam_next_idx+1]-lam[lam_next_idx])+curve_js[lam_next_idx+1]*(lam[lam_next_idx+1]-lam_next)/(lam[lam_next_idx+1]-lam[lam_next_idx])

	# 		curve_js_cmd=np.zeros(len(robot_state.joint_position))
	# 		np.put(curve_js_cmd,(~ctrl_joints.astype(bool)).astype(int).nonzero()[0],q_static)
	# 		if len(joint_recording)==0:
	# 			np.put(curve_js_cmd,ctrl_joints.nonzero()[0],qd_next)
	# 		else:
	# 			np.put(curve_js_cmd,ctrl_joints.nonzero()[0],qd_next+(qd_cur-q_cur)*0.1)

	# 		# print(curve_js_cmd)
	# 		ts,js=self.position_cmd(curve_js_cmd,now)
	# 		timestamp_recording.append(ts)
	# 		joint_recording.append(js[ctrl_joints.nonzero()[0]])
	# 		q_cur=js[ctrl_joints.nonzero()[0]]

	# 	timestamp_recording=np.array(timestamp_recording)
	# 	timestamp_recording-=timestamp_recording[0]
	# 	return timestamp_recording, np.array(joint_recording)
	

	def add_extension_egm_js(self,lower_limit,upper_limit,curve_cmd_js,extension_start=50,extension_end=50):
		#################add extension#########################
		init_extension_js=np.linspace(curve_cmd_js[0]-extension_start*(curve_cmd_js[1]-curve_cmd_js[0]),curve_cmd_js[0],num=extension_start,endpoint=False)
		end_extension_js=np.linspace(curve_cmd_js[-1],curve_cmd_js[-1]+extension_end*(curve_cmd_js[-1]-curve_cmd_js[-2]),num=extension_end+1)[1:]

		###cap extension within joint limits
		for i in range(len(curve_cmd_js[0])):
			init_extension_js[:,i]=np.clip(init_extension_js[:,i],lower_limit[i]+0.01,upper_limit[i]-0.01)
			end_extension_js[:,i]=np.clip(end_extension_js[:,i],lower_limit[i]+0.01,upper_limit[i]-0.01)

		curve_cmd_js_ext=np.vstack((init_extension_js,curve_cmd_js,end_extension_js))
		return curve_cmd_js_ext