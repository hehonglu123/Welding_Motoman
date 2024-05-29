import cv2, time, os
import pickle, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../../toolbox/')
from flir_toolbox import *
from robot_def import *

# Load the IR recording data from the pickle file
data_dir='../../../recorded_data/weld_scan_job205_v152023_07_27_13_23_06/'
config_dir='../../config/'

robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15)
robot2=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
	pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
	pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose_mocap.csv')


cv2.namedWindow("IR Recording", cv2.WINDOW_NORMAL)
cmap = cv2.COLORMAP_INFERNO
pixel2mm=1
image_center=(240/2,320/2)
timeoffset=3


#find all folders
folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
centroids_all=[]
spatial_centroids_all=[]

for folder in folders:
    with open(data_dir+folder+'/ir_recording.pickle', 'rb') as file:
        ir_recording = pickle.load(file)
    ir_ts=np.loadtxt(data_dir+folder+'/ir_stamps.csv', delimiter=',')
    try:
        ir_ts=(ir_ts-ir_ts[0])
    except IndexError:
        print(folder+' has no IR data')
        continue
    joint_angle=np.loadtxt(data_dir+folder+'/joint_recording.csv', delimiter=',')
    spatial_centroids_layer=[]

    for i in range(len(ir_recording)):
        #rotate image by 90 degrees, keeping the scale
        ir_image = np.rot90(ir_recording[i], k=-1)
        # ir_image = ir_recording[i]

        centroid, bbox=flame_detection(ir_image)
        if centroid is not None:
            centroids_all.append(centroid)

            #find the correct corresponding time in joint recording
            idx=np.argmin(np.abs(ir_ts[i]-joint_angle[:,0]-timeoffset))
            pose2=robot2.fwd(joint_angle[idx][8:-2])
            spatial_centroid=[pose2.p[0]+pixel2mm*(centroid[0]-image_center[0]),pose2.p[2]+pixel2mm*(image_center[1]-centroid[1])]
            spatial_centroids_layer.append(spatial_centroid)

       
        
        # ir_normalized = ((ir_image - np.min(ir_image)) / (np.max(ir_image) - np.min(ir_image))) * 255
        # ir_normalized=np.clip(ir_normalized, 0, 255)       
        # ir_bgr = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)
        # # Display the IR image
        # cv2.resizeWindow("IR Recording", 240, 320)
        # cv2.imshow("IR Recording", ir_bgr)
        # # Wait for a specific time (in milliseconds) before displaying the next frame
        # cv2.waitKey(1)

    spatial_centroids_layer=np.array(spatial_centroids_layer)
    spatial_centroids_all.append(spatial_centroids_layer)
    plt.plot(spatial_centroids_layer[:,0], spatial_centroids_layer[:,1], 'o')    
    plt.title('Layer Height from Flame Tracking (PoC)')


#plot as 2d line
plt.show()
