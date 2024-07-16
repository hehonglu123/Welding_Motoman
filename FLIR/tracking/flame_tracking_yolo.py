import pickle, sys
import numpy as np
from flir_toolbox import *
from ultralytics import YOLO

yolo_model = YOLO("yolov8/torch.pt")

# data_dir='../../../recorded_data/ER316L/wallbf_70ipm_v7_70ipm_v7/'
# data_dir='../../../recorded_data/ER316L/cylinderspiral_100ipm_v10/'
data_dir='../../../recorded_data/ER316L/streaming/cylinderspiral_T19000/'

with open(data_dir+'/ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)


frame=8752
ir_image = np.rot90(ir_recording[frame], k=-1)
centroid, bbox, torch_centroid, torch_bbox=flame_detection_yolo(ir_image,yolo_model,percentage_threshold=0.8)
