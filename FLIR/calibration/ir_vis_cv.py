import cv2
import pickle, sys
import numpy as np
from thermal_couple_conversion import voltage_to_temperature

sys.path.append('../toolbox/')
# from flir_toolbox import *

room_temp=20

# Load the IR recording data from the pickle file
ir_recording=np.load('recorded_data/ir_images.npy')[:]
temperature_reading=np.loadtxt('recorded_data/temperature_reading.csv',delimiter=',')

temperature_bias=room_temp-voltage_to_temperature(np.average(temperature_reading[:10,1]))

# Create a window to display the images
cv2.namedWindow("IR Recording", cv2.WINDOW_NORMAL)
# Set the colormap (inferno) and normalization range for the color bar
cmap = cv2.COLORMAP_INFERNO
colorbar_min = np.min(ir_recording)
colorbar_max = np.max(ir_recording)

for i in range(len(ir_recording)):
    print(i)
    ir_normalized = ((ir_recording[i] - np.min(ir_recording[i])) / (colorbar_max-colorbar_min)) * 255

    # ir_normalized = ir_normalized[50:-50, 50:-50]
    ir_normalized=np.clip(ir_normalized, 0, 255)

    # Convert the IR image to BGR format with the inferno colormap
    ir_bgr = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)

    # add temperature
    cv2.putText(ir_bgr, '%.2fC'% (voltage_to_temperature(temperature_reading[i,1])+temperature_bias), (50, 50), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (0, 255, 0) , 2, cv2.LINE_AA) 
    # Display the IR image
    cv2.imshow("IR Recording", ir_bgr)

    # Wait for a specific time (in milliseconds) before displaying the next frame
    cv2.waitKey(100)

# Close the window after the loop is completed
cv2.destroyAllWindows()
