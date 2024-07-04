
from scipy.interpolate import interp1d
import asyncio, pickle, base64, threading, copy, time, traceback
from motoman_robotraconteur_driver.motoplus_rr_driver_feedback_client import MotoPlusRRDriverFeedbackSyncClient
from motoman_robotraconteur_driver.motoplus_rr_driver_command_client import MotoPlusRRDriverCommandClient, StreamingMotionTarget
from motoman_robotraconteur_driver.motoplus_rr_driver_streaming_command_client import MotoplusRRDriverStreamingCommandClient
import numpy as np
from contextlib import suppress
import traceback


class MotoplusStreaming:
	async def start(self, IP='192.168.1.31'):
		##################Joint Streaming############################
		self.streaming_rate=125
		c = MotoPlusRRDriverCommandClient()
		self.c_udp = MotoplusRRDriverStreamingCommandClient()
		c.start(IP)        
		self.c_udp.start(IP)
		await c.wait_ready(10)
		
		s='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'
		self.controller_info=pickle.loads(base64.b64decode(s))

		with suppress(Exception):
			await c.stop_motion_streaming()
		await c.start_motion_streaming()

		##################Joint Feedback Readming############################
		self.fb = MotoPlusRRDriverFeedbackSyncClient()
		self.fb.start(IP)
		self._recording=False
		self.joint_recording=[]
		self._lock=threading.Lock()

	def read_joint(self):
		res, fb_data = self.fb.try_receive_state_sync(self.controller_info, 0.01)
		if res:
			joint_angle=np.hstack((fb_data.group_state[0].command_position ,fb_data.group_state[1].command_position ,fb_data.group_state[2].command_position ))
			timestamp=fb_data.time
			return True, timestamp, joint_angle, fb_data.job_state
		else:
			return False, None, None, None
		
	def threadfunc(self):
		while(self._recording):
			try:         
				res, timestamp, self.joint_angle, job_state = self.read_joint()
				if res:
					self.joint_recording.append(np.array([timestamp]+self.joint_angle.tolist()+[job_state[0][1],job_state[0][2]]))
				else:
					pass
					# print("Failed to read state")
			except:
				traceback.print_exc()
	
	def StartRecording(self):
		if self._recording:     ###if already streaming
			return
		self._recording=True
		self.joint_recording=[]
		t=threading.Thread(target=self.threadfunc)
		t.daemon=True
		t.start()

	def StopRecording(self):
		self._recording=False
		return np.array(self.joint_recording)


	def position_cmd(self,qd,start_time):
		target = [
			StreamingMotionTarget(0, np.multiply(qd[:6], self.controller_info.control_groups[0].pulse_to_radians)),
			StreamingMotionTarget(1, np.multiply(qd[6:12], self.controller_info.control_groups[1].pulse_to_radians)),
			StreamingMotionTarget(2, np.multiply(qd[12:14], self.controller_info.control_groups[2].pulse_to_radians)),
		]
		self.c_udp.send_motion_streaming_pulse_target(target)
		# while time.time()-start_time<1/self.streaming_rate-0.001:
		# 	continue
		time.sleep(float(1.0/self.streaming_rate)-0.001)

	def jog2q(self,qd,point_distance=0.2):
		q_cur=copy.deepcopy(self.joint_angle)
		num_points_jogging=self.streaming_rate*np.max(np.abs(q_cur-qd))/point_distance

		for j in range(int(num_points_jogging)):
			q_target = (q_cur*(num_points_jogging-j))/num_points_jogging+qd*j/num_points_jogging
			self.position_cmd(q_target,time.time())
			
		###init point wait
		for i in range(20):
			self.position_cmd(qd,time.time())
		self.joint_angle = qd

async def main():
	MS=MotoplusStreaming()
	await MS.start(IP='127.0.0.1')
	MS.StartRecording()
	while True:
		try:
			print('runnnig')
			MS.jog2q(np.hstack((np.zeros(6),[np.pi/2,0,0,0,0,0,np.radians(-15),np.pi])))
			MS.jog2q(np.hstack((-0.5*np.ones(6),[np.pi/2,0,0,0,0,0,np.radians(-15),np.pi])))
		except:
			js_recording = MS.StopRecording()
			traceback.print_exc()
			break

	np.savetxt('joint_recording_streaming.csv',js_recording,delimiter=',')

if __name__ == '__main__':
	asyncio.run(main())