import cv2
import pickle, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../../toolbox/')
from flir_toolbox import *

def normalize2cv(frame):
    ir_normalized = ((frame - np.min(frame)) / (np.max(frame) - np.min(frame))) * 255
    return ir_normalized.astype(np.uint8)


# Load the IR recording data from the pickle file
#70S_model_120ipm_2023_09_23_21_27_03
#316L_model_120ipm_2023_09_25_19_56_43
with open('../../../recorded_data/316L_model_120ipm_2023_09_25_19_56_43/layer_3/ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)

ir_ts=np.loadtxt('../../../recorded_data/316L_model_120ipm_2023_09_25_19_56_43/layer_3/ir_stamps.csv', delimiter=',')

plt.plot(ir_ts, ir_recording[:,172,88])
# plt.imshow(ir_recording[0], cmap='inferno')
plt.title('Pixel Value vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Pixel Value (Counts)')
plt.show()

result = cv2.VideoWriter('output.avi', 
                         cv2.VideoWriter_fourcc(*'XVID'),
                         30, (320,240))

# Create a window to display the images
cv2.namedWindow("IR Flow", cv2.WINDOW_NORMAL)
# Set the colormap (inferno) and normalization range for the color bar
cmap = cv2.COLORMAP_INFERNO
colorbar_min = np.min(ir_recording)
colorbar_max = np.max(ir_recording)

for i in range(1,len(ir_recording)):
    prev=ir_recording[i-1]
    cur=ir_recording[i]
    # Compute the frame difference
    # diff = cv2.absdiff(prev, cur)
    diff=cur.astype(np.double)-prev.astype(np.double)
    print(np.max(diff),np.min(diff))
    # Optionally apply a colormap to the difference
    ir_colormap = cv2.applyColorMap(normalize2cv(ir_recording[i]), cv2.COLORMAP_INFERNO)
    ir_colormap[172,88]=[0,255,0]
    # result.write(ir_colormap)
    cv2.imshow("IR Flow",ir_colormap)
    # Wait for a specific time (in milliseconds) before displaying the next frame
    cv2.waitKey(int(1000*(ir_ts[i+1]-ir_ts[i])))

result.release()
# Close the window after the loop is completed
cv2.destroyAllWindows()
