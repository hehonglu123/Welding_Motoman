import cv2
import pickle, sys
import numpy as np
sys.path.append('../toolbox/')
from flir_toolbox import *

# Load the IR recording data from the pickle file
data_dir='../../recorded_data/ER316L/streaming/right_triangle/video_bf_ol_v10_f100'
with open(data_dir+'/ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)
ir_ts=np.loadtxt(data_dir+'/ir_stamps.csv', delimiter=',')

# Set the colormap (inferno) and normalization range for the color bar
cmap = cv2.COLORMAP_INFERNO

# Define the video writer
output_file = 'ir_recording_video.mp4'
frame_height, frame_width = ir_recording[0].shape
fps = 30  # Frames per second
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Loop through each frame and write to the video
for i in range(len(ir_recording)):
    ir_image = np.rot90(ir_recording[i], k=-1)

    # Normalize the IR image
    ir_normalized = ((ir_image - np.min(ir_image)) / (np.max(ir_image) - np.min(ir_image))) * 255
    ir_normalized = np.clip(ir_normalized, 0, 255)

    # Convert the IR image to BGR format with the inferno colormap
    ir_bgr = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)

    # Ensure the frame size is consistent
    if ir_bgr.shape[0] != frame_height or ir_bgr.shape[1] != frame_width:
        ir_bgr = cv2.resize(ir_bgr, (frame_width, frame_height))

    # Write the frame to the video
    out.write(ir_bgr)

    # Display the timestamp in Terminal
    print('\rTimeStamp: %.5f' % (ir_ts[i] - ir_ts[0]), end='', flush=True)

# Release the video writer
out.release()

print("\nVideo saved as", output_file)