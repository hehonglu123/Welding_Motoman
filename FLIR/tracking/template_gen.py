import cv2
import pickle, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../../toolbox/')
from flir_toolbox import *

data_dir='../../../recorded_data/ER316L/wallbf_70ipm_v7_70ipm_v7/'
with open(data_dir+'/ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)

frame=10000
ir_image = np.rot90(ir_recording[frame], k=-1)
pixel_threshold = 1.0e4
ir_image[ir_image>pixel_threshold]=pixel_threshold
print(np.max(ir_image))
ir_normalized = ((ir_image - np.min(ir_image)) / (np.max(ir_image) - np.min(ir_image)))
ir_normalized=np.clip(ir_normalized, 0, 255)

###display raw image to identify the torch visually
plt.imshow(ir_normalized)
plt.show()
###test the visual bounding box
upper_left=(115,95)
lower_right=(160,145)
plt.imshow(ir_normalized[upper_left[1]:lower_right[1],upper_left[0]:lower_right[0]])
plt.show()



# normalized to [0, 255] in unin8 format
image = cv2.normalize(ir_normalized[upper_left[1]:lower_right[1],upper_left[0]:lower_right[0]], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


# Apply Canny edge detection, more aggressive
edges = cv2.Canny(image, threshold1=50, threshold2=130)

# save template as binary image
cv2.imwrite('torch_template_temp.png',edges)

# Display the original image and the edge detected image side by side
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()