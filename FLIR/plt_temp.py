import cv2,wave,copy, yaml
import pickle, sys
import numpy as np
import scipy.signal as signal
sys.path.append('../toolbox/')
from flir_toolbox import *
import matplotlib.pyplot as plt

data_dir="../../recorded_data/weld_scan_job205_v152023_07_27_13_23_06/layer_150/"
with open('calibration.yml', 'r') as file:
	param= yaml.safe_load(file)
print(param)
# Load the IR recording data from the pickle file
with open(data_dir+'ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)
ir_ts=np.loadtxt(data_dir+'ir_stamps.csv', delimiter=',')
ir_ts=ir_ts-ir_ts[0]

temp_below=counts2temp(ir_recording[:,130:140,165:175].flatten(),param[0.13][0], param[0.13][1], param[0.13][2], param[0.13][3], param[0.13][4],Emiss=0.13).reshape((-1,100))
temp_avg=np.average(temp_below, axis=1)
plt.plot(ir_ts, temp_avg, c='red', label='temperature below')
plt.title('Average Temperature Beneath')
plt.ylabel('Temperature (C)')
plt.legend()
plt.show()
