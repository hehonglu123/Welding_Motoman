import cv2,copy
import pickle, os, inspect
import numpy as np
import matplotlib.pyplot as plt
from flir_toolbox import *
from motoman_def import *
from ultralytics import YOLO


#load template
template = cv2.imread('../tracking/torch_template_ER316L.png',0)

# Load the IR recording data from the pickle file
# data_dir='../../../recorded_data/ER316L_IR_wall_study/wallbf_150ipm_v15_150ipm_v15/'
data_dir='../../../recorded_data/ER316L/trianglebf_100ipm_v10_100ipm_v10/'
# data_dir='../../../recorded_data/wallbf_100ipm_v10_80ipm_v8/'
# data_dir='../../../recorded_data/wallbf_100ipm_v10_120ipm_v12/'
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

#load model
torch_model = YOLO(os.path.dirname(inspect.getfile(flir_toolbox))+"/torch.pt")
tip_wire_model = YOLO(os.path.dirname(inspect.getfile(flir_toolbox))+"/tip_wire.pt")

vertical_offset=3
horizontal_offset=0

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



for layer_num in range(20,len(layer_indices_ir)-1):
    pixel_coord_layer=[]    #find all pixel regions to record from flame detection
    #find all pixel regions to record from flame detection
    for i in range(layer_indices_ir[layer_num],layer_indices_ir[layer_num+1]):
        ir_image = np.rot90(ir_recording[i], k=-1)
        # centroid, bbox=flame_detection(ir_image,threshold=1.0e4,area_threshold=10)

        centroid, bbox, torch_centroid, torch_bbox=weld_detection_steel(ir_image,torch_model,tip_wire_model)
        if centroid is not None:
            #find average pixel value 
            pixel_coord = (int(centroid[0]) + horizontal_offset, int(centroid[1]) + vertical_offset)
            pixel_coord_layer.append(pixel_coord)

    pixel_coord_layer=np.array(pixel_coord_layer)
    #remove duplicate pixel regions
    pixel_coord_layer=np.unique(pixel_coord_layer,axis=0)
    #smoothout the pixel regions with running average
    pixel_coord_layer[:,1]=moving_average(pixel_coord_layer[:,1],3,padding=True)

    #go over again for the identified pixel regions value
    ts_all=[]
    pixel_all=[]
    counts_all=[]
    for i in range(layer_indices_ir[layer_num],layer_indices_ir[layer_num+1]):
        ir_image = np.rot90(ir_recording[i], k=-1)
        ts_all.extend([ir_ts[i]]*len(pixel_coord_layer))
        pixel_all.extend(pixel_coord_layer[:,0])
        for coord in pixel_coord_layer:
            #find the NxN ir_pixel_window_size average pixel value below centroid
            counts_all.append(get_pixel_value(ir_image,coord,ir_pixel_window_size))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_trisurf(ts_all, pixel_all, counts_all, linewidth=0, antialiased=False, label='-')

    plt.title('Pixel Value vs Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Column/X (pixel)')
    ax.set_zlabel('Pixel Value (Counts)')
    plt.show()