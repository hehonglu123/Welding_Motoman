import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = np.load('torch_template.npy')
# normalized to [0, 255] in unin8 format
image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


# Apply Canny edge detection
edges = cv2.Canny(image, threshold1=50, threshold2=200)

# save template as binary image
cv2.imwrite('torch_template.png',edges)

# Display the original image and the edge detected image side by side
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()