import cv2
import pickle, sys
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
sys.path.append('../../toolbox/')
from flir_toolbox import *

def normalize2cv(frame):
    ir_normalized = ((frame - np.min(frame)) / (np.max(frame) - np.min(frame))) * 255
    return ir_normalized.astype(np.uint8)

for i in range(2,10):
    # Load the IR recording data from the pickle file
    with open('../../../recorded_data/316L_model_120ipm_2023_09_25_19_56_43/layer_%i/ir_recording.pickle'%(i), 'rb') as file:
        ir_recording = pickle.load(file)
    data = read_csv('../../../recorded_data/316L_model_120ipm_2023_09_25_19_56_43/layer_%i/command1.csv'%(i))
    v=data['weld_v'].tolist()[-1][1:-1]
    ir_ts=np.loadtxt('../../../recorded_data/316L_model_120ipm_2023_09_25_19_56_43/layer_%i/ir_stamps.csv'%(i), delimiter=',')

    #canny edge detection
    ir_normalized=normalize2cv(ir_recording[0])
    edged = cv2.Canny(ir_normalized, 100, 200)
    indices=np.argwhere(edged[:,88]==255)
    if len(indices)==0:
        print("no edge detected")
    for index in indices:
        if ir_recording[0][index+3,88]>100:
            break
    # plt.imshow(edged)

    plt.plot(ir_ts, ir_recording[:,index+3,88],label='v='+v+'mm/s')

plt.title('Pixel Value vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Pixel Value (Counts)')
plt.legend()
plt.show()


