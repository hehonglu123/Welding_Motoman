import cv2,wave,copy
import pickle, sys
import numpy as np
import scipy.signal as signal
sys.path.append('../toolbox/')
from flir_toolbox import *
import matplotlib.pyplot as plt

data_dir="../../recorded_data/weld_scan_job205_v152023_07_27_13_23_06/layer_150/"

# Load the IR recording data from the pickle file
with open(data_dir+'ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)
ir_ts=np.loadtxt(data_dir+'ir_stamps.csv', delimiter=',')
ir_ts=ir_ts-ir_ts[0]
center_brightness=np.average(ir_recording[:,110:140,150:180],axis=(1,2))
all_birghtness=np.average(ir_recording,axis=(1,2))
plt.plot(ir_ts, center_brightness, c='red', label='ir center counts')
plt.plot(ir_ts, all_birghtness, c='blue', label='ir all counts')
plt.title('Average IR Pixel Counts ')
plt.ylabel('Center Brightness (counts)')
plt.xlabel('Time (s)')
plt.legend()
plt.show()
