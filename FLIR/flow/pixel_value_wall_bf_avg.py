import cv2,copy
import pickle, sys
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
sys.path.append('../../toolbox/')
from flir_toolbox import *
from robot_def import *


def center_of_window_below_bbox(bbox,ir_pixel_window_size, num_pixel_below_centroid=0):
    # Calculate the bottom center point of the bbox
    x, y, w, h = bbox

    center_x = int(x + w/2) 
    center_y = y + max(h+ir_pixel_window_size//2, h//2+num_pixel_below_centroid)

    return center_x, center_y

#load template
template = cv2.imread('../tracking/torch_template_ER316L.png',0)

# Load the IR recording data from the pickle file
data_dir='../../../recorded_data/ER316L_IR_wall_study/wallbf_70ipm_v7_70ipm_v7/'
config_dir='../../config/'
with open(data_dir+'/ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)
ir_ts=np.loadtxt(data_dir+'/ir_stamps.csv', delimiter=',')
joint_angle=np.loadtxt(data_dir+'weld_js_exe.csv',delimiter=',')

ir_pixel_window_size=7

robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15)
robot2=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
	pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
	pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose_mocap.csv')

#find indices of each layer by detecting velocity direction change
p_all=robot.fwd(joint_angle[:,2:8]).p_all
v_all=np.gradient(p_all,axis=0)/np.gradient(joint_angle[:,0])[:,None]
v_direction=v_all@(p_all[len(p_all)//2]-p_all[len(p_all)//2-1])
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
for layer_num in range(20,len(layer_indices_ir)-1):
    #find all pixel regions to record from flame detection
    for i in range(layer_indices_ir[layer_num],layer_indices_ir[layer_num+1]):
        ir_image = np.rot90(ir_recording[i], k=-1)
        # centroid, bbox=flame_detection(ir_image,threshold=1.0e4,area_threshold=10)
        centroid, bbox=flame_detection_no_arc(ir_image,template)
        if centroid is not None:
            #find 3x3 average pixel value below centroid
            pixel_coord=center_of_window_below_bbox(bbox,ir_pixel_window_size)
            pixel_value_all.append(get_pixel_value(ir_image,pixel_coord,ir_pixel_window_size))
            ir_ts_processed.append(ir_ts[i])

print(np.mean(pixel_value_all))
plt.title('Pixel Value vs Time (10 layers)')
plt.plot(ir_ts_processed, pixel_value_all)
plt.xlabel('Time (s)')
plt.ylabel('Pixel Value')
plt.show()