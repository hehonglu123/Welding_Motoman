import cv2
import pickle, sys
import numpy as np
sys.path.append('../toolbox/')
# from flir_toolbox import *


# Load the IR recording data from the pickle file
ir_recording=np.load('recorded_data/ir_images.npy')[:]

print(len(ir_recording))

result = cv2.VideoWriter('output.avi', 
                         cv2.VideoWriter_fourcc(*'XVID'),
                         30, (320,240))

# Create a window to display the images
cv2.namedWindow("IR Recording", cv2.WINDOW_NORMAL)
# Set the colormap (inferno) and normalization range for the color bar
cmap = cv2.COLORMAP_INFERNO
colorbar_min = np.min(ir_recording)
colorbar_max = np.max(ir_recording)

for i in range(len(ir_recording)):
    print(i)
    ir_normalized = ((ir_recording[i] - np.min(ir_recording[i])) / (np.max(ir_recording[i]) - np.min(ir_recording[i]))) * 255

    # ir_normalized = ir_normalized[50:-50, 50:-50]
    ir_normalized=np.clip(ir_normalized, 0, 255)

    # Convert the IR image to BGR format with the inferno colormap
    ir_bgr = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)

    # Display the IR image
    cv2.imshow("IR Recording", ir_bgr)

    # Wait for a specific time (in milliseconds) before displaying the next frame
    cv2.waitKey(100)

result.release()
# Close the window after the loop is completed
cv2.destroyAllWindows()
