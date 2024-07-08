import cv2
import pickle, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../../toolbox/')
from flir_toolbox import *

#load template
template = cv2.imread('torch_template_ER316L.png',0)
scale_factor=1.0
template.resize((int(scale_factor*template.shape[0]),int(scale_factor*template.shape[1])))
# Load the IR recording data from the pickle file
# with open('../../../recorded_data/ER316L_wall_streaming_bf/layer_394/ir_recording.pickle', 'rb') as file:
#     ir_recording = pickle.load(file)

# data_dir='../../../recorded_data/ER316L/wallbf_70ipm_v7_70ipm_v7/'
data_dir='../../../recorded_data/ER316L/cylinderspiral_100ipm_v10/'

with open(data_dir+'/ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)


# Create a window to display the images
cv2.namedWindow("IR Recording", cv2.WINDOW_NORMAL)
# Set the colormap (inferno) and normalization range for the color bar
cmap = cv2.COLORMAP_INFERNO
colorbar_min = np.min(ir_recording)
colorbar_max = np.max(ir_recording)


frame=10000
ir_image = np.rot90(ir_recording[frame], k=-1)
centroid, bbox=flame_detection_no_arc(ir_image,template,threshold=2.0e4)

