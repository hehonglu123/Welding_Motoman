import ultralytics
from ultralytics import YOLO
import cv2
import pickle, sys
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
import torch_tracking, inspect, os


if __name__ == "__main__":
    
    # Load the IR recording data from the pickle file
    # data_dir='../../../../recorded_data/ER316L/wallbf_100ipm_v10_100ipm_v10/'
    # data_dir='../../../../recorded_data/ER316L/streaming/cylinderspiral_T22222/'
    data_dir='../../../../recorded_data/ER4043/wallbf_100ipm_v10_70ipm_v7/'


    with open(data_dir+'/ir_recording.pickle', 'rb') as file:
        ir_recording = pickle.load(file)
    ir_ts=np.loadtxt(data_dir+'/ir_stamps.csv', delimiter=',')

    # Set the colormap (inferno) and normalization range for the color bar
    cmap = cv2.COLORMAP_INFERNO

    frame=10000
    pixel_threshold=1e4
    ir_torch_tracking = np.rot90(ir_recording[frame], k=-1)
    ir_torch_tracking[ir_torch_tracking>pixel_threshold]=pixel_threshold
    ir_torch_tracking_normalized = ((ir_torch_tracking - np.min(ir_torch_tracking)) / (np.max(ir_torch_tracking) - np.min(ir_torch_tracking))) * 255
    ir_torch_tracking_normalized = ir_torch_tracking_normalized.astype(np.uint8)
    ir_torch_tracking = cv2.cvtColor(ir_torch_tracking_normalized, cv2.COLOR_GRAY2BGR)

    #run yolo
    model = YOLO("torch.pt")
    # results = yolo(ir_torch_tracking_normalized)
    # results= model.predict("datasets/torch_yolov8_data/train/images/30_png.rf.0971a2c2d86bf76299a739b019758aa0.jpg",imgsz=320)
    results= model.predict(ir_torch_tracking,conf=0.5)
    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        #convert to numpy
        # print(result.boxes.cls.cpu().numpy())
        # print(boxes.cpu().xyxy[0].numpy())
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen