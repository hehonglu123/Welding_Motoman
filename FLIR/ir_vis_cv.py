import cv2
import pickle
import numpy as np

# Load the IR recording data from the pickle file
with open('recorded_data/slice_120_0_flir.pickle', 'rb') as file:
    ir_recording = pickle.load(file)

ir_ts=np.loadtxt('recorded_data/slice_120_0_flir_ts.csv', delimiter=',')

# Create a window to display the images
cv2.namedWindow("IR Recording", cv2.WINDOW_NORMAL)
# Set the colormap (inferno) and normalization range for the color bar
cmap = cv2.COLORMAP_INFERNO
colorbar_min = np.min(ir_recording)
colorbar_max = np.max(ir_recording)

for i in range(len(ir_recording)):
    # print(np.max(ir_recording[i]), np.min(ir_recording[i]))

    # Normalize the data to [0, 255]
    ir_normalized = ((ir_recording[i] - np.min(ir_recording[i])) / (np.max(ir_recording[i]) - np.min(ir_recording[i]))) * 255

    # ir_normalized = ir_normalized[50:-50, 50:-50]
    ir_normalized=np.clip(ir_normalized, 0, 255)

    # Convert the IR image to BGR format with the inferno colormap
    ir_bgr = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)

    # # Create a color bar image using the inferno colormap
    # color_bar = np.arange(colorbar_min, colorbar_max, (colorbar_max - colorbar_min) / 100).reshape(100, 1)
    # color_bar_normalized = ((color_bar - colorbar_min) / (colorbar_max - colorbar_min)) * 255
    # color_bar_bgr = cv2.applyColorMap(color_bar_normalized[::-1].astype(np.uint8), cmap)
    # color_bar_image = cv2.resize(color_bar_bgr, (50, ir_bgr.shape[0]))
    # ir_bgr=np.hstack((ir_bgr, color_bar_image))

    # Display the IR image
    cv2.imshow("IR Recording", ir_bgr)

    # Wait for a specific time (in milliseconds) before displaying the next frame
    cv2.waitKey(int(1000*(ir_ts[i+1]-ir_ts[i])))

# Close the window after the loop is completed
cv2.destroyAllWindows()
