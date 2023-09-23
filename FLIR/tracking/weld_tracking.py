import cv2, time
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


result = cv2.VideoWriter('output.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         13, (320,240))

# Create a window to display the images
cv2.namedWindow("IR Recording", cv2.WINDOW_NORMAL)
# Set the colormap (inferno) and normalization range for the color bar
cmap = cv2.COLORMAP_INFERNO
colorbar_min = np.min(ir_recording)
colorbar_max = np.max(ir_recording)

moving_direction=0  #0: left, 1: right
centroid_prev=[160,120]
for i in range(len(ir_recording)):
    # print(np.max(ir_recording[i]), np.min(ir_recording[i]))
    now=time.time()
    centroid, bbox, pixels=weld_detection(ir_recording[i])
    # print(time.time()-now)

    temp=counts2temp(ir_recording[i].flatten(),6.39661118e+03, 1.40469989e+03, 1.00000008e+00, 8.69393436e+00, 8.40029488e+03,Emiss=0.8).reshape((240,320))
    
    # temp[temp > 1300] = 1300    ##thresholding
    # temp=ir_recording[i]
    # Normalize the data to [0, 255]
    ir_normalized = ((temp - np.min(temp)) / (np.max(temp) - np.min(temp))) * 255
    
    # ir_normalized = ir_normalized[50:-50, 50:-50]
    ir_normalized=np.clip(ir_normalized, 0, 255).astype(np.uint8)

    # Convert the IR image to BGR format with the inferno colormap
    ir_bgr = cv2.applyColorMap(ir_normalized, cv2.COLORMAP_INFERNO)

    #torch detection
    torch_loc=torch_detect(ir_recording[i],template)
    # add bounding box
    cv2.rectangle(ir_bgr, torch_loc, (torch_loc[0] + template.shape[1], torch_loc[1] + template.shape[0]), (0,255,0), 2)
    torch_eef=[torch_loc[0]+template.shape[1]-3,torch_loc[1]+int(template.shape[0]/2)]
    # add bounding box
    if centroid is not None:
        indices=[np.argmax(pixels[0]),np.argmin(pixels[0])]
        idx=indices[np.argmin(np.abs(pixels[0][indices]-torch_eef[1]))]
        tip=[pixels[1][idx],pixels[0][idx]] #x,y coordinates
        
        cv2.line(ir_bgr,tuple(torch_eef),(int(tip[0]),int(tip[1])),(0,255,0))
        ir_bgr[pixels[0],pixels[1],:]=[0,255,0]



    # Write the IR image to the video file
    result.write(ir_bgr)

    # Display the IR image
    # cv2.imshow("IR Recording", ir_bgr)

    # # Wait for a specific time (in milliseconds) before displaying the next frame
    # cv2.waitKey(int(1000*(ir_ts[i+1]-ir_ts[i])))

result.release()
# Close the window after the loop is completed
cv2.destroyAllWindows()
