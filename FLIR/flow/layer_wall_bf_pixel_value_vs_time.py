import cv2, os, pickle, inspect, time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from flir_toolbox import *
from motoman_def import *
from ultralytics import YOLO


# Load the IR recording data from the pickle file
# data_dir='../../../recorded_data/ER316L_IR_wall_study/wallbf_70ipm_v7_70ipm_v7/'
# data_dir='../../../recorded_data/ER316L/trianglebf_100ipm_v10_100ipm_v10/'
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

#find indices of each layer by detecting velocity direction change
p_all=robot.fwd(joint_angle[:,2:8]).p_all
v_all=np.gradient(p_all,axis=0)/np.gradient(joint_angle[:,0])[:,None]
v_direction=v_all@(p_all[len(p_all)//2]-p_all[len(p_all)//2-2])
signs=np.sign(np.sign(v_direction)+0.1)

layer_indices_js=[0]
layer_indices_ir=[0]
sign_cur=signs[0]
sign_continuous_time=0
for i in range(1,len(signs)):
    if signs[i]==sign_cur:
        sign_continuous_time=joint_angle[i,0]-joint_angle[layer_indices_js[-1],0]
    else:
        if sign_continuous_time>2:  #remain same direction for at least 2sec
            layer_indices_js.append(i)
            layer_indices_ir.append(np.argmin(np.abs(ir_ts-joint_angle[i,0])))
        sign_cur=signs[i]
        sign_continuous_time=0


pixel_value_all=[]
ir_ts_processed=[]
frame_processing_time=[]
for layer_num in tqdm(range(len(layer_indices_ir)-1)):
    #find all pixel regions to record from flame detection
    for i in range(layer_indices_ir[layer_num],layer_indices_ir[layer_num+1]):
        start_time=time.time()
        ir_image = np.rot90(ir_recording[i], k=-1)
        centroid, bbox, torch_centroid, torch_bbox=weld_detection_steel(ir_image,torch_model,tip_wire_model)
        if centroid is not None:
            #find 3x3 average pixel value below centroid
            pixel_coord = (int(centroid[0]) + horizontal_offset, int(centroid[1]) + vertical_offset)
            pixel_value_all.append(get_pixel_value(ir_image,pixel_coord,ir_pixel_window_size))
            ir_ts_processed.append(ir_ts[i])

            frame_processing_time.append(time.time()-start_time)

pixel_value_all=moving_average(pixel_value_all,n=31,padding=True)
ir_ts_processed=np.array(ir_ts_processed)-ir_ts_processed[0]
plt.title('Pixel Value vs Time ')
plt.plot(ir_ts_processed, pixel_value_all)
plt.xlabel('Time (s)')
plt.ylabel('Pixel Value')
plt.show()