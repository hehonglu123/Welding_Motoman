import cv2, os, pickle, inspect, time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from flir_toolbox import *
from motoman_def import *
from ultralytics import YOLO


# Load the IR recording data from the pickle file
data_dir='../../../recorded_data/ER316L/streaming/right_triangle/bf_T25000/'
# data_dir='../../../recorded_data/ER316L/streaming/wall2/bf_ol_v10_f100/'
# data_dir='../../../recorded_data/ER316L/streaming/wall2/bf_T25000/'

config_dir='../../config/'
with open(data_dir+'/ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)
ir_ts=np.loadtxt(data_dir+'/ir_stamps.csv', delimiter=',')
joint_angle=np.loadtxt(data_dir+'weld_js_exe.csv',delimiter=',')

ir_pixel_window_size=7

robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15)

#load model
torch_model = YOLO(os.path.dirname(inspect.getfile(flir_toolbox))+"/torch.pt")
tip_wire_model = YOLO(os.path.dirname(inspect.getfile(flir_toolbox))+"/tip_wire.pt")

vertical_offset=3
horizontal_offset=0
#find idling indices of
idle_time_interval=[]
idling=False
idle_counts=0
for i in range(len(joint_angle)-1):
    if np.linalg.norm(joint_angle[i+1][2:8]-joint_angle[i][2:8])<1e-6:
        idle_counts+=1
        if not idling:
            idling=True
            idle_start_time=joint_angle[i][0]
    else:
        if idling:
            idling=False
            if idle_counts>250:
                idle_stop_time=joint_angle[i][0]
                idle_time_interval.append([idle_start_time,idle_stop_time])
        idle_counts=0

#throw away reading within idle time
throw_away_idx=[]
for i in range(len(idle_time_interval)):
    throw_away_idx.append(np.where((ir_ts>=idle_time_interval[i][0]) & (ir_ts<=idle_time_interval[i][1]))[0])

throw_away_idx=np.concatenate(throw_away_idx)

# ir_ts=np.delete(ir_ts,throw_away_idx)
ir_recording=np.delete(ir_recording,throw_away_idx,axis=0)
ir_ts_new=np.linspace(0,len(ir_recording)/30,len(ir_recording))


pixel_value_all=[]
ir_ts_processed=[]
flame_centroid_history=[]

for i in tqdm(range(len(ir_recording))):
    ir_image = np.rot90(ir_recording[i], k=-1)
    centroid, bbox, torch_centroid, torch_bbox=weld_detection_steel(ir_image,torch_model,tip_wire_model)
    if centroid is not None:
        ###weighted history filter
        if len(flame_centroid_history) > 30:
            flame_centroid_history.pop(0)
            # Calculate the weight for the previous history values
            previous_weight = 0.8 / len(flame_centroid_history)
            centroid = 0.2 * centroid + np.sum(np.array(flame_centroid_history) * previous_weight, axis=0)
            flame_centroid_history.append(centroid)
                        
        #find 3x3 average pixel value below centroid
        pixel_coord = (int(centroid[0]) + horizontal_offset, int(centroid[1]) + vertical_offset)
        pixel_value_all.append(get_pixel_value(ir_image,pixel_coord,ir_pixel_window_size))
        ir_ts_processed.append(ir_ts_new[i])

###filtered value for every 30 frames
pixel_value_all_processed=[]
frames_per_process=30
ir_ts_processed_new=[]
for i in range(frames_per_process,len(pixel_value_all)):
    if i%frames_per_process==0:
        pixel_value_all_processed.append(np.mean(pixel_value_all[i-frames_per_process:i]))
        ir_ts_processed_new.append(ir_ts_processed[i])

pixel_value_all=moving_average(pixel_value_all,n=31,padding=True)
plt.title('Pixel Value vs Time ')
plt.plot(ir_ts_processed_new, pixel_value_all_processed)
plt.xlabel('Time (s)')
plt.ylabel('Pixel Value')
plt.show()