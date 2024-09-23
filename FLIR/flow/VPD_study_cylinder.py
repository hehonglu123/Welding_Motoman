from tqdm import tqdm
import pickle, os, inspect
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from flir_toolbox import *
from motoman_def import *
from ultralytics import YOLO

#load model
torch_model = YOLO(os.path.dirname(inspect.getfile(flir_toolbox))+"/torch.pt")
tip_wire_model = YOLO(os.path.dirname(inspect.getfile(flir_toolbox))+"/tip_wire.pt")

VPD=10
vertical_offset=3
horizontal_offset=0

pixel_value_avg_all=[]
pixel_value_std_all=[]
v_all=np.arange(5,21)
for v in tqdm(v_all):
    # Load the IR recording data from the pickle file
    data_dir='../../../recorded_data/ER316L/VPD%i/tubespiral_%iipm_v%i/'%(VPD,VPD*v,v)

    config_dir='../../config/'
    with open(data_dir+'/ir_recording.pickle', 'rb') as file:
        ir_recording = pickle.load(file)
    ir_ts=np.loadtxt(data_dir+'/ir_stamps.csv', delimiter=',')
    joint_angle=np.loadtxt(data_dir+'weld_js_exe.csv',delimiter=',')


    frame_start=5*len(ir_recording)//20     ###get rid of the first 5 layers
    frame_end=len(ir_recording)
    # frame_end=20000
    ir_pixel_window_size=7

    pixel_value_all=[]
    ir_ts_processed=[]
    flame_centroid_history = []
    for i in tqdm(range(frame_start, frame_end)):
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
                                    
            #find average pixel value 
            pixel_coord = (int(centroid[0]) + horizontal_offset, int(centroid[1]) + vertical_offset)
            flame_reading=get_pixel_value(ir_image,pixel_coord,ir_pixel_window_size)
            pixel_value_all.append(flame_reading)
            ir_ts_processed.append(ir_ts[i])


    pixel_value_avg_all.append(np.mean(pixel_value_all))
    pixel_value_std_all.append(np.std(pixel_value_all))
    print("Average pixel value: ", np.mean(pixel_value_all))
    plt.title('Pixel Value vs Time ')
    plt.plot(ir_ts_processed, pixel_value_all)
    #plot the red line at the average pixel value
    plt.axhline(y=np.mean(pixel_value_all), color='r', linestyle='-',label='Average Pixel Value')
    plt.xlabel('Time (s)')
    plt.ylabel('Pixel Value')
    plt.ylim(15000,28000)
    plt.savefig('pixel_value_vs_time_%iipm_v%i.png'%(VPD*v,v))
    plt.clf()


plt.errorbar(v_all, pixel_value_avg_all, yerr=pixel_value_std_all, fmt='-o', capsize=5)
plt.xlabel('Speed (mm/s)')
plt.ylabel('Pixel Value')
plt.title('Pixel Value for 20 layers')
plt.show()