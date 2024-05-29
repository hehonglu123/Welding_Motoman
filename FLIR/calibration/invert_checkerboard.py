import cv2

# Read the image
img = cv2.imread('pattern.png', cv2.IMREAD_GRAYSCALE)

# Invert the image
img_inverted = cv2.bitwise_not(img)

cv2.imwrite('pattern_inverted.png', img_inverted)