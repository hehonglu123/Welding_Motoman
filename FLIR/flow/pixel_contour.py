import cv2
import pickle, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../../toolbox/')
from flir_toolbox import *

def normalize2cv(frame):
    ir_normalized = ((frame - np.min(frame)) / (np.max(frame) - np.min(frame))) * 255
    return ir_normalized.astype(np.uint8)

VPD=20
vertical_offset=3
horizontal_offset=0
v=15
# Load the IR recording data from the pickle file
data_dir='../../../recorded_data/ER316L/phi0.9_VPD20/cylinderspiral_%iipm_v%i/'%(VPD*v,v)

config_dir='../../config/'
with open(data_dir+'/ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)
ir_ts=np.loadtxt(data_dir+'/ir_stamps.csv', delimiter=',')
joint_angle=np.loadtxt(data_dir+'weld_js_exe.csv',delimiter=',')



# Set the colormap (inferno) and normalization range for the color bar
cmap = cv2.COLORMAP_INFERNO
colorbar_min = np.min(ir_recording)
colorbar_max = np.max(ir_recording)
cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)

for i in range(1000, len(ir_recording)):
    # Normalize the image to 8-bit for OpenCV
    normalized_img = cv2.normalize(ir_recording[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Create a binary mask where pixel values are above 25000
    threshold = 0.9*np.max(ir_recording[i])
    binary_mask = (ir_recording[i] > threshold).astype(np.uint8) * 255
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    # Apply colormap to the normalized image
    colored_img = cv2.applyColorMap(normalized_img, cmap)

    # Draw contours on the normalized image
    cv2.drawContours(colored_img, contours, -1, (0, 255, 0), 1, lineType=cv2.LINE_8)
    
    
    
    # Display the image with contours using OpenCV window
    cv2.imshow("Contours", colored_img)
    # Wait for a key press and break the loop if 'q' is pressed
    cv2.waitKey(int(1000*(ir_ts[i+1]-ir_ts[i])))

# Destroy all OpenCV windows
cv2.destroyAllWindows()

