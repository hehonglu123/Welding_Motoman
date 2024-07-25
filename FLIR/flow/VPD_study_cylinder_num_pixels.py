from tqdm import tqdm
import pickle, os, inspect
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from flir_toolbox import *
from motoman_def import *
from ultralytics import YOLO

def get_pixel_value_new(ir_image,coord,window_size):
    ###get pixel value larger than avg within the window
    window = ir_image[coord[1]-window_size//2:coord[1]+window_size//2+1,coord[0]-window_size//2:coord[0]+window_size//2+1]
    pixel_avg = np.mean(window)
    mask = (window > pixel_avg) # filter out background 
    pixel_avg = np.mean(window[mask])
    return pixel_avg

#load model
torch_model = YOLO(os.path.dirname(inspect.getfile(flir_toolbox))+"/torch.pt")
tip_wire_model = YOLO(os.path.dirname(inspect.getfile(flir_toolbox))+"/tip_wire.pt")

VPD=10
for v in tqdm(range(5,14)):
    # Load the IR recording data from the pickle file
    data_dir='../../../recorded_data/ER316L/VPD10/cylinderspiral_%iipm_v%i/'%(VPD*v,v)

    config_dir='../../config/'
    with open(data_dir+'/ir_recording.pickle', 'rb') as file:
        ir_recording = pickle.load(file)
    ir_ts=np.loadtxt(data_dir+'/ir_stamps.csv', delimiter=',')
    joint_angle=np.loadtxt(data_dir+'weld_js_exe.csv',delimiter=',')


    frame_start=0
    frame_end=len(ir_recording)
    # frame_end=20000
    ir_pixel_window_size=5

    num_pixels_history = []
    pixel_threshold = 20000     #count num pixels above this threshold
    for i in tqdm(range(frame_start, frame_end)):
        ir_image = np.rot90(ir_recording[i], k=-1)
        num_pixels_history.append(np.sum(ir_image>pixel_threshold))



    print("Number of pixels above threshold: ", np.mean(num_pixels_history))
    plt.title('Pixel Value vs Time ')
    plt.plot(ir_ts, num_pixels_history)
    #plot the red line at the average pixel value
    plt.axhline(y=np.mean(num_pixels_history), color='r', linestyle='-',label='Average Pixel Value')
    plt.xlabel('Time (s)')
    plt.ylabel('Number of Pixels')
    plt.savefig('num_pixel_counts_vs_time_%iipm_v%i.png'%(VPD*v,v))
    plt.clf()