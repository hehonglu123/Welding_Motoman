import cv2
import pickle, sys
import numpy as np
sys.path.append('../../toolbox/')
from flir_toolbox import *


# Load the IR recording data from the pickle file
torch_template=np.load('torch_template.npy')



# Create a window to display the images
cv2.namedWindow("IR Recording", cv2.WINDOW_NORMAL)
# Set the colormap (inferno) and normalization range for the color bar
cmap = cv2.COLORMAP_INFERNO

print(torch_template.shape)

temp=counts2temp(torch_template.flatten(),6.39661118e+03, 1.40469989e+03, 1.00000008e+00, 8.69393436e+00, 8.40029488e+03,Emiss=0.13).reshape(torch_template.shape)
temp[temp > 1300] = 1300    ##thresholding
# Normalize the data to [0, 255]
ir_normalized = ((temp - np.min(temp)) / (np.max(temp) - np.min(temp))) * 255

# ir_normalized = ir_normalized[50:-50, 50:-50]
ir_normalized=np.clip(ir_normalized, 0, 255)

# Convert the IR image to BGR format with the inferno colormap
ir_bgr = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)



# Display the IR image
cv2.imshow("IR Recording", ir_bgr)


cv2.waitKey()
# Close the window after the loop is completed
cv2.destroyAllWindows()
