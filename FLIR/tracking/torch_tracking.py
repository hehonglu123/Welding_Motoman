import cv2
import pickle, sys
import numpy as np
sys.path.append('../../toolbox/')
from flir_toolbox import *

#load template
template = cv2.imread('torch_template.png',0)

# Load the IR recording data from the pickle file
with open('../../../recorded_data/ER316L_wall_streaming_bf/layer_394/ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)

ir_ts=np.loadtxt('../../../recorded_data/ER316L_wall_streaming_bf/layer_394/ir_stamps.csv', delimiter=',')


# Create a window to display the images
cv2.namedWindow("IR Recording", cv2.WINDOW_NORMAL)
# Set the colormap (inferno) and normalization range for the color bar
cmap = cv2.COLORMAP_INFERNO
colorbar_min = np.min(ir_recording)
colorbar_max = np.max(ir_recording)


frame=308

temp=counts2temp(ir_recording[frame].flatten(),6.39661118e+03, 1.40469989e+03, 1.00000008e+00, 8.69393436e+00, 8.40029488e+03,Emiss=0.13).reshape((240,320))
temp[temp > 1300] = 1300    ##thresholding

ir_normalized = ((temp - np.min(temp)) / (np.max(temp) - np.min(temp))) * 255

max_loc=torch_detect(ir_recording[frame],template)


# Convert the IR image to BGR format with the inferno colormap
ir_bgr = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)


# add bounding box
cv2.rectangle(ir_bgr, max_loc, (max_loc[0] + template.shape[1], max_loc[1] + template.shape[0]), (0,255,0), 2)


# Display the IR image
cv2.imshow("IR Recording", ir_bgr)


cv2.waitKey()
# Close the window after the loop is completed
cv2.destroyAllWindows()

np.save('torch_template.npy',ir_normalized[110:175,40:135])
