import RobotRaconteur as RR
RRN=RR.RobotRaconteurNode.s
import time, copy, sys
import numpy as np
sys.path.append('../toolbox/flir_toolbox')
from matplotlib import pyplot as plt
from ultralytics import YOLO

ir_process="""
service experimental.ir_process

object ir_process_obj
	wire uint16 flame_reading [readonly]
	wire uint16[] torch_bottom [readonly]
	wire uint16[] weld_pool [readonly]
end object
"""


class FLIR_RR_Process(object):
	def __init__(self,flir_service, yolo_model):
		
		self.flir_service=flir_service
		self.ir_image_consts = RRN.GetConstants('com.robotraconteur.image', self.flir_service)
		self.flir_service.setf_param("focus_pos", RR.VarValue(int(1900),"int32"))
		self.flir_service.setf_param("object_distance", RR.VarValue(0.4,"double"))
		self.flir_service.setf_param("reflected_temperature", RR.VarValue(291.15,"double"))
		self.flir_service.setf_param("atmospheric_temperature", RR.VarValue(293.15,"double"))
		self.flir_service.setf_param("relative_humidity", RR.VarValue(50,"double"))
		self.flir_service.setf_param("ext_optics_temperature", RR.VarValue(293.15,"double"))
		self.flir_service.setf_param("ext_optics_transmission", RR.VarValue(0.99,"double"))
		self.flir_service.setf_param("current_case", RR.VarValue(2,"int32"))
		self.flir_service.setf_param("ir_format", RR.VarValue("radiometric","string"))
		self.flir_service.setf_param("object_emissivity", RR.VarValue(0.13,"double"))
		self.flir_service.setf_param("scale_limit_low", RR.VarValue(293.15,"double"))
		self.flir_service.setf_param("scale_limit_upper", RR.VarValue(5000,"double"))
		self.cam_pipe=self.flir_service.frame_stream.Connect(-1)
		#Set the callback for new pipe packets
		self.start_ir_cb = False
		self.cam_pipe.PacketReceivedEvent+=self.ir_cb
		try:
			self.flir_service.start_streaming()
		except:
			pass
		
		#processing parameters
		self.ir_pixel_window_size=7
		self.yolo_model = yolo_model
		
	
			
	def ir_cb(self,pipe_ep):
		# Loop to get the newest frame
		while (pipe_ep.Available > 0):
			# Receive the packet
			rr_img = pipe_ep.ReceivePacket()
			if self.start_ir_cb:
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

				ir_image = np.rot90(display_mat, k=-1)
				centroid, bbox, torch_centroid, torch_bbox=flame_detection_yolo(ir_image,self.yolo_model,percentage_threshold=0.8)
				if centroid is not None:
					#find 3x3 average pixel value below centroid
					center_x = centroid[0]
					center_y = centroid[1]+self.ir_pixel_window_size//2
					pixel_coord=(center_x,center_y)
					flame_reading=get_pixel_value(ir_image,pixel_coord,self.ir_pixel_window_size)

					self.flame_reading.OutValue=int(flame_reading)		#flame pixel value
					self.torch_bottom.OutValue=torch_bbox.astype(int)	#torch bottom center pixel coordinates
					self.weld_pool.OutValue=centroid.astype(int)		#weld pool pixel coordinates


def main():
	yolo_model = YOLO("../tracking/yolov8/torch.pt")
	
	with RR.ServerNodeSetup("experimental.ir_process", 12182):
		flir_service=RRN.ConnectService('rr+tcp://localhost:60827/?service=camera')
		#Register the service type
		RRN.RegisterServiceType(ir_process)

		ir_process_obj=FLIR_RR_Process(flir_service, yolo_model)
		
		#Register the service
		RRN.RegisterService("FLIR_RR_PROCESS","experimental.ir_process.ir_process_obj",ir_process_obj)
		input("Press enter to quit")
	
if __name__ == '__main__':
	main()