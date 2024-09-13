import RobotRaconteur as RR
RRN=RR.RobotRaconteurNode.s
import numpy as np
from flir_toolbox import *
from ultralytics import YOLO
import torch_tracking, os, inspect, traceback, pickle, time

ir_process="""
service experimental.ir_process
struct ir_process_struct
	field uint16 flame_reading
	field uint16[] torch_bottom
	field uint16[] arc_centroid
end 
object ir_process_obj
	wire ir_process_struct ir_process_result [readonly]
end object
"""


class FLIR_RR_Process(object):
	def __init__(self, yolo_model):
		
		#processing parameters
		self.ir_pixel_window_size=7
		self.yolo_model = yolo_model
		self.ir_process_struct=RRN.NewStructure("experimental.ir_process.ir_process_struct")
		self.flame_centroid_history = []
	
			
	def playback(self,ir_ts,ir_recording):
		#sort indices by timestamp
		sort_idx=np.argsort(ir_ts)
		for i in sort_idx:

			ir_image = np.rot90(ir_recording[i], k=-1)
			centroid, bbox, torch_centroid, torch_bbox=weld_detection_steel(ir_image,self.yolo_model,percentage_threshold=0.77)
			if centroid is not None:
				###weighted history filter
				if len(self.flame_centroid_history) > 30:
					self.flame_centroid_history.pop(0)
					# Calculate the weight for the previous history values
					previous_weight = 0.8 / len(self.flame_centroid_history)
					centroid = 0.2 * centroid + np.sum(np.array(self.flame_centroid_history) * previous_weight, axis=0)
					self.flame_centroid_history.append(centroid)

				pixel_coord=(centroid[0],centroid[1]+5)
				flame_reading=get_pixel_value(ir_image,pixel_coord,self.ir_pixel_window_size)

				print(flame_reading, torch_bbox, centroid)
				try:
					self.ir_process_struct.flame_reading=int(flame_reading)
					self.ir_process_struct.torch_bottom=np.array([torch_bbox[0]+torch_bbox[2]//2, torch_bbox[1]+torch_bbox[3]]).astype(np.uint16)
					self.ir_process_struct.weld_pool=centroid.astype(np.uint16)
					self.ir_process_result.OutValue=self.ir_process_struct
				except:
					traceback.print_exc()
				
			time.sleep(1./30.)

				


def main():
	yolo_model = YOLO(os.path.dirname(inspect.getfile(torch_tracking))+"/torch.pt")
	data_dir='../../../recorded_data/ER316L/streaming/cylinderspiral_T22222/'
	with open(data_dir+'/ir_recording.pickle', 'rb') as file:
		ir_recording = pickle.load(file)
	ir_ts=np.loadtxt(data_dir+'/ir_stamps.csv', delimiter=',')
	
	with RR.ServerNodeSetup("experimental.ir_process", 12182):
		#Register the service type
		RRN.RegisterServiceType(ir_process)

		flir_playback_service=FLIR_RR_Process(yolo_model)
		
		#Register the service
		RRN.RegisterService("FLIR_RR_PROCESS","experimental.ir_process.ir_process_obj",flir_playback_service)
		flir_playback_service.playback(ir_ts,ir_recording)
	
if __name__ == '__main__':
	main()