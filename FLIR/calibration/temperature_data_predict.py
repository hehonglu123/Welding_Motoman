import numpy as np 
import pickle, cv2
import matplotlib.pyplot as plt

calibration=np.load('ER4043_IR_calibration.npy')


# Load the IR recording data from the pickle file
with open('../../../recorded_data/weld_scan_job205_v152023_07_27_13_23_06/layer_300/ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)

ir_ts=np.loadtxt('../../../recorded_data/weld_scan_job205_v152023_07_27_13_23_06/layer_300/ir_stamps.csv', delimiter=',')

colorbar_min = np.min(ir_recording)
colorbar_max = 10000#np.max(ir_recording)

for i in range(len(ir_recording)):
    
    ir_normalized = ((ir_recording[i] - np.min(ir_recording[i])) / (colorbar_max-colorbar_min)) * 255
    ir_bgr = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)
    # Display the IR image
    cv2.imshow("IR Recording", ir_bgr)

    # Wait for a specific time (in milliseconds) before displaying the next frame
    cv2.waitKey(100)
    # plt.imshow(ir_bgr)
    # plt.show()