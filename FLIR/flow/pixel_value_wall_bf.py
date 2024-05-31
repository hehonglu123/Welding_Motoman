import cv2,copy
import pickle, sys
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
sys.path.append('../../toolbox/')
from flir_toolbox import *


def center_of_window_below_bbox(bbox):
    # Calculate the bottom center point of the bbox
    x, y, w, h = bbox
    center_bottom = (x + w // 2, y + h)

    # Define the 3x3 region below the bbox
    start_x = center_bottom[0] - 1
    start_y = center_bottom[1]
    end_x = start_x + 3
    end_y = start_y + 3

    # Calculate the center of the 3x3 window
    center_x = (start_x + end_x) // 2
    center_y = (start_y + end_y) // 2

    return center_x, center_y

# Load the IR recording data from the pickle file
data_dir='../../../recorded_data/wall_bf_100ipm_v10/'
config_dir='../../config/'
with open(data_dir+'/ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)
ir_ts=np.loadtxt(data_dir+'/ir_stamps.csv', delimiter=',')



###Identify layers first
timeslot=[229,239,249,259,269,279,289,299,309,319,329,339,349,359]
duration=10
for start_time in timeslot:
    start_idx=np.argmin(np.abs(ir_ts-ir_ts[0]-start_time))
    end_idx=np.argmin(np.abs(ir_ts-ir_ts[0]-start_time-duration))
    pixel_coord_layer=[]    #find all pixel regions to record from flame detection
    #find all pixel regions to record from flame detection
    for i in range(start_idx,end_idx):
        ir_image = np.rot90(ir_recording[i], k=-1)
        centroid, bbox=flame_detection(ir_image,threshold=1.0e4,area_threshold=10)
        if centroid is not None:
            #find 3x3 average pixel value below centroid
            pixel_coord=center_of_window_below_bbox(bbox)
            pixel_coord_layer.append(pixel_coord)
    pixel_coord_layer=np.array(pixel_coord_layer)

    #go over again for the identified pixel regions value
    ts_all=[]
    pixel_all=[]
    counts_all=[]
    for i in range(start_idx,end_idx):
        ir_image = np.rot90(ir_recording[i], k=-1)
        ts_all.extend([ir_ts[i]]*len(pixel_coord_layer))
        pixel_all.extend(pixel_coord_layer[:,0])
        for coord in pixel_coord_layer:
            #find the 3x3 average pixel value below centroid
            counts_all.append(np.mean(ir_image[coord[1]-1:coord[1]+2,coord[0]-1:coord[0]+2]))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_trisurf(ts_all, pixel_all, counts_all, linewidth=0, antialiased=False, label='-')

    plt.title('Pixel Value vs Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Column/X (pixel)')
    ax.set_zlabel('Pixel Value (Counts)')
    plt.show()