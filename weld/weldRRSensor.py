from RobotRaconteur.Client import *
import time
import numpy as np
import copy
import pickle
import wave
from matplotlib import pyplot as plt

class WeldRRSensor(object):
	def __init__(self,weld_service=None,\
				cam_service=None,\
				microphone_service=None,\
				current_service=None) -> None:
		
		## weld service
		self.weld_service=weld_service
		if weld_service:
			self.weld_obj = self.weld_service.GetDefaultClientWait(3)  # connect, timeout=30s
			self.welder_state_sub = self.weld_service.SubscribeWire("welder_state")
			self.start_weld_cb = False
			self.clean_weld_record()
			self.welder_state_sub.WireValueChanged += self.weld_cb
		
		## IR Camera Service
		self.cam_ser=cam_service
		if cam_service:
			self.ir_image_consts = RRN.GetConstants('com.robotraconteur.image', self.cam_ser)

			self.cam_ser.setf_param("focus_pos", RR.VarValue(int(1600),"int32"))
			self.cam_ser.setf_param("object_distance", RR.VarValue(0.4,"double"))
			self.cam_ser.setf_param("reflected_temperature", RR.VarValue(291.15,"double"))
			self.cam_ser.setf_param("atmospheric_temperature", RR.VarValue(293.15,"double"))
			self.cam_ser.setf_param("relative_humidity", RR.VarValue(50,"double"))
			self.cam_ser.setf_param("ext_optics_temperature", RR.VarValue(293.15,"double"))
			self.cam_ser.setf_param("ext_optics_transmission", RR.VarValue(0.99,"double"))
			self.cam_ser.setf_param("current_case", RR.VarValue(2,"int32"))
			self.cam_ser.setf_param("ir_format", RR.VarValue("radiometric","string"))
			self.cam_ser.setf_param("object_emissivity", RR.VarValue(0.13,"double"))
			self.cam_ser.setf_param("scale_limit_low", RR.VarValue(293.15,"double"))
			self.cam_ser.setf_param("scale_limit_upper", RR.VarValue(5000,"double"))

			self.cam_pipe=self.cam_ser.frame_stream.Connect(-1)
			#Set the callback for new pipe packets
			self.start_ir_cb = False
			self.cam_pipe.PacketReceivedEvent+=self.ir_cb
			try:
				self.cam_ser.start_streaming()
			except:
				pass
			self.clean_ir_record()
		
		## microphone service
		self.mic_service=microphone_service
		if microphone_service:
			self.mic_samplerate = 44100
			self.mic_channels = 1
			self.mic_service=microphone_service
			self.mic_pipe = self.mic_service.microphone_stream.Connect(-1)
			self.clean_mic_record()
			self.start_mic_cb=False
			self.mic_pipe.PacketReceivedEvent+=self.microphone_cb

		self.current_service=current_service
		if current_service:
			self.current_state_sub = self.current_service.SubscribeWire("current")
			self.start_current_cb = False
			self.clean_current_record()
			self.current_state_sub.WireValueChanged += self.current_cb

	def start_all_sensors(self):

		if self.weld_service:
			self.clean_weld_record()
			self.start_weld_cb=True
		if self.cam_ser:
			self.clean_ir_record()
			self.start_ir_cb=True
		if self.mic_service:
			self.clean_mic_record()
			self.start_mic_cb=True
		if self.current_service:
			self.clean_current_record()
			self.start_current_cb=True
	
	def clear_all_sensors(self):
		if self.weld_service:
			self.clean_weld_record()
		if self.cam_ser:
			self.clean_ir_record()
		if self.mic_service:
			self.clean_mic_record()
		if self.current_service:
			self.clean_current_record()

	def stop_all_sensors(self):

		if self.weld_service:
			self.start_weld_cb=False
			self.weld_timestamp=np.array(self.weld_timestamp)
		if self.cam_ser:
			self.start_ir_cb=False
			self.ir_timestamp=np.array(self.ir_timestamp)
		if self.mic_service:
			self.start_mic_cb=False
		if self.current_service:
			self.start_current_cb=False
			self.current_timestamp=np.array(self.current_timestamp)
	
	def save_all_sensors(self,filedir):

		if self.weld_service:
			self.save_weld_file(filedir)
		if self.cam_ser:
			self.save_ir_file(filedir)
		if self.mic_service:
			self.save_mic_file(filedir)
		if self.current_service:
			self.save_current_file(filedir)

	
	def test_all_sensors(self,t=3):

		self.start_all_sensors()
		time.sleep(t)
		self.stop_all_sensors()

		if self.cam_ser:
			fig = plt.figure(1)
			sleep_t=float(3./len(self.ir_recording))
			for r in self.ir_recording:
				plt.imshow(r, cmap='inferno', aspect='auto')
				plt.colorbar(format='%.2f')
				plt.pause(sleep_t)
				plt.clf()
		if self.mic_service:
			first_channel = np.concatenate(self.audio_recording)
			first_channel_int16=(first_channel*32767).astype(np.int16)
			plt.plot(first_channel_int16)
			plt.title("Microphone data")
			plt.show()
		if self.current_service:
			print("Current data length:",len(self.current))
			plt.plot(self.current_timestamp,self.current)
			plt.title("Current data")
			plt.show()
	
	def clean_weld_record(self):

		self.weld_timestamp=[]
		self.weld_voltage=[]
		self.weld_current=[]
		self.weld_feedrate=[]
		self.weld_energy=[]

	def clean_current_record(self):
		self.current=[]
		self.current_timestamp=[]

	def weld_cb(self, sub, value, ts):

		if self.start_weld_cb:
			self.weld_timestamp.append(value.ts['microseconds'][0])
			self.weld_voltage.append(value.welding_voltage)
			self.weld_current.append(value.welding_current)
			self.weld_feedrate.append(value.wire_speed)
			self.weld_energy.append(value.welding_energy)
	
	def current_cb(self, sub, value, ts):

		if self.start_current_cb:
			self.current_timestamp.append(ts.seconds+ts.nanoseconds*1e-9)
			self.current.append(value)

	
	def save_weld_file(self,filedir):
		np.savetxt(filedir + 'welding.csv',
				np.array([(self.weld_timestamp-self.weld_timestamp[0])/1e6, self.weld_voltage, self.weld_current, self.weld_feedrate, self.weld_energy]).T, delimiter=',',
				header='timestamp,voltage,current,feedrate,energy', comments='')
		
	def save_current_file(self,filedir):
		np.savetxt(filedir + 'current.csv',
				np.array([(self.current_timestamp-self.current_timestamp[0]), self.current]).T, delimiter=',',
				header='timestamp,current', comments='')
	
	def clean_ir_record(self):
		self.ir_timestamp=[]
		self.ir_recording=[]

	def ir_cb(self,pipe_ep):

		# Loop to get the newest frame
		while (pipe_ep.Available > 0):
			# Receive the packet
			rr_img = pipe_ep.ReceivePacket()
			if not self.start_ir_cb:
				continue
			if rr_img.image_info.encoding == self.ir_image_consts["ImageEncoding"]["mono8"]:
				# Simple uint8 image
				mat = rr_img.data.reshape([rr_img.image_info.height, rr_img.image_info.width], order='C')
			elif rr_img.image_info.encoding == self.ir_image_consts["ImageEncoding"]["mono16"]:
				data_u16 = np.array(rr_img.data.view(np.uint16))
				mat = data_u16.reshape([rr_img.image_info.height, rr_img.image_info.width], order='C')

			ir_format = rr_img.image_info.extended["ir_format"].data

			if ir_format == "temperature_linear_10mK":
				display_mat = (mat * 0.01) - 273.15
			elif ir_format == "temperature_linear_100mK":
				display_mat = (mat * 0.1) - 273.15
			else:
				display_mat = mat

			# Convert the packet to an image and set the global variable
			self.ir_recording.append(copy.deepcopy(display_mat))
			self.ir_timestamp.append(rr_img.image_info.data_header.ts['seconds']+rr_img.image_info.data_header.ts['nanoseconds']*1e-9)
	
	def save_ir_file(self,filedir):

		with open(filedir+'ir_recording.pickle','wb') as file:
				pickle.dump(np.array(self.ir_recording),file)
		np.savetxt(filedir + "ir_stamps.csv",self.ir_timestamp-self.ir_timestamp[0],delimiter=',')
	
	def clean_mic_record(self):

		self.audio_recording=[]
	
	def microphone_cb(self,pipe_ep):

		#Loop to get the newest frame
		while (pipe_ep.Available > 0):
			audio = pipe_ep.ReceivePacket().audio_data
			if not self.start_mic_cb:
				continue
			#Receive the packet
			self.audio_recording.extend(audio)
	
	def save_mic_file(self,filedir):

		# print("Mic length:",len(self.audio_recording))

		try:
			first_channel = np.concatenate(self.audio_recording)

			first_channel_int16=(first_channel*32767).astype(np.int16)
			with wave.open(filedir+'mic_recording.wav', 'wb') as wav_file:
				# Set the WAV file parameters
				wav_file.setnchannels(self.mic_channels)
				wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
				wav_file.setframerate(self.mic_samplerate)

				# Write the audio data to the WAV file
				wav_file.writeframes(first_channel_int16.tobytes())
		except:
			print("Mic has no recording!!!")
