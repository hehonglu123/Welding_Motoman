import cv2
import pickle, sys
import numpy as np
sys.path.append('../toolbox/')
from flir_toolbox import *

# Load the IR recording data from the pickle file
with open('../../recorded_data/weld_scan_job205_v152023_07_27_13_23_06/layer_150/ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)

ir_ts=np.loadtxt('../../recorded_data/weld_scan_job205_v152023_07_27_13_23_06/layer_150/ir_stamps.csv', delimiter=',')


result = cv2.VideoWriter('output.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         13, (320,240),0)

# Create a window to display the images
cv2.namedWindow("IR Recording", cv2.WINDOW_NORMAL)
# Set the colormap (inferno) and normalization range for the color bar
cmap = cv2.COLORMAP_INFERNO
colorbar_min = np.min(ir_recording)
colorbar_max = np.max(ir_recording)

for i in range(len(ir_recording)):
    # print(np.max(ir_recording[i]), np.min(ir_recording[i]))
    temp=counts2temp(ir_recording[i].flatten(),6.39661118e+03, 1.40469989e+03, 1.00000008e+00, 8.69393436e+00, 8.40029488e+03,Emiss=0.13).reshape((240,320))
    temp[temp > 1300] = 1300    ##thresholding
    temp[temp < 200] = 200    ##thresholding
    
    # Normalize the data to [0, 255]
    ir_normalized = ((temp - np.min(temp)) / (np.max(temp) - np.min(temp))) * 255

    # ir_normalized = ir_normalized[50:-50, 50:-50]
    ir_normalized=np.clip(ir_normalized, 0, 255).astype(np.uint8)

    ir_edge=cv2.Canny(ir_normalized, 55, 255, 15)
    # Convert the IR image to BGR format with the inferno colormap
    ir_bgr = cv2.applyColorMap(ir_normalized, cv2.COLORMAP_INFERNO)

    # Write the IR image to the video file
    # result.write(ir_edge)

    # Display the IR image
    cv2.imshow("IR Recording", ir_edge)

    # Wait for a specific time (in milliseconds) before displaying the next frame
    cv2.waitKey(int(1000*(ir_ts[i+1]-ir_ts[i])))

result.release()
# Close the window after the loop is completed
cv2.destroyAllWindows()
