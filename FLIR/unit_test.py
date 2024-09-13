# Capture and display streaming frames

from RobotRaconteur.Client import *
import numpy as np
import matplotlib.pyplot as plt
import time, traceback, pickle

image_data=[]

def new_frame(pipe_ep):
	global image_data
	#Loop to get the newest frame
	while (pipe_ep.Available > 0):
		#Receive the packet
		rr_img=pipe_ep.ReceivePacket()
		if rr_img.image_info.encoding == image_consts["ImageEncoding"]["mono8"]:
			# Simple uint8 image
			mat = rr_img.data.reshape([rr_img.image_info.height, rr_img.image_info.width], order='C')
		elif rr_img.image_info.encoding == image_consts["ImageEncoding"]["mono16"]:
			data_u16 = np.array(rr_img.data.view(np.uint16))
			mat = data_u16.reshape([rr_img.image_info.height, rr_img.image_info.width], order='C')
		
		ir_format = rr_img.image_info.extended["ir_format"].data

		if ir_format == "temperature_linear_10mK":
			display_mat = (mat * 0.01) - 273.15    
		elif ir_format == "temperature_linear_100mK":
			display_mat = (mat * 0.1) - 273.15    
		else:
			display_mat = mat

		#Convert the packet to an image and set the global variable
		ts=rr_img.image_info.data_header.ts['seconds']+rr_img.image_info.data_header.ts['nanoseconds']*1e-9
		image_data.append(display_mat)
		



def main():
	global image_data
	now=time.time()
	url='rr+tcp://192.168.55.10:60827/?service=camera'

	c1=RRN.ConnectService(url)
	c1.setf_param("focus_pos", RR.VarValue(int(1900),"int32"))
	c1.setf_param("object_distance", RR.VarValue(0.3,"double"))
	c1.setf_param("reflected_temperature", RR.VarValue(291.15,"double"))
	c1.setf_param("atmospheric_temperature", RR.VarValue(293.15,"double"))
	c1.setf_param("relative_humidity", RR.VarValue(50,"double"))
	c1.setf_param("ext_optics_temperature", RR.VarValue(293.15,"double"))
	c1.setf_param("ext_optics_transmission", RR.VarValue(0.99,"double"))
	c1.setf_param("current_case", RR.VarValue(2,"int32"))
	c1.setf_param("ir_format", RR.VarValue("radiometric","string"))
	c1.setf_param("object_emissivity", RR.VarValue(0.9,"double"))
	c1.setf_param("scale_limit_low", RR.VarValue(293.15,"double"))
	c1.setf_param("scale_limit_upper", RR.VarValue(5000,"double"))

	global image_consts, ts
	ts=0
	image_consts = RRN.GetConstants('com.robotraconteur.image', c1)

	p=c1.frame_stream.Connect(-1)

	#Set the callback for when a new pipe packet is received to the
	#new_frame function
	p.PacketReceivedEvent+=new_frame
	try:
		c1.start_streaming()
	except: pass


	fig = plt.figure(1)
	
	while True:
		time.sleep(60)
		try:
			with open('ir_recording.pickle','wb') as file:
				pickle.dump(np.array(image_data),file)
				image_data=[]
		except:
			traceback.print_exc()
		finally:
			break



		

if __name__ == "__main__":
	main()