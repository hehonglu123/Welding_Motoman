import cv2,wave,copy, yaml
import pickle, sys
import numpy as np
import scipy.signal as signal
sys.path.append('../toolbox/')
from flir_toolbox import *
import matplotlib.pyplot as plt

data_dir='../data/wall_weld_test/moveL_100_repeat_weld_scan_2023_08_02_17_07_02/layer_16/'
with open('calibration.yml', 'r') as file:
	param= yaml.safe_load(file)
print(param)
# Load the IR recording data from the pickle file
with open(data_dir+'ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)
ir_ts=np.loadtxt(data_dir+'ir_stamps.csv', delimiter=',')
ir_ts=ir_ts-ir_ts[0]

# temp_below=counts2temp(ir_recording[:,130:140,165:175].flatten(),param[0.13][0], param[0.13][1], param[0.13][2], param[0.13][3], param[0.13][4],Emiss=0.13).reshape((-1,100))
# temp_avg=np.average(temp_below, axis=1)

###active bounding box tracking
temp_below=[]
for i in range(len(ir_recording)):
    # print(np.max(ir_recording[i]), np.min(ir_recording[i]))
    centroid, bbox=flame_detection(ir_recording[i])
    temp=counts2temp(ir_recording[i].flatten(),6.39661118e+03, 1.40469989e+03, 1.00000008e+00, 8.69393436e+00, 8.40029488e+03,Emiss=0.13).reshape((240,320))
    if centroid is not None:
        bbox_below_size=10
        centroid_below=(int(centroid[0]+bbox[2]/2+bbox_below_size/2),centroid[1])
        temp_below.append(np.average(temp[int(centroid_below[1]-bbox_below_size):int(centroid_below[1]+bbox_below_size),int(centroid_below[0]-bbox_below_size):int(centroid_below[0]+bbox_below_size)]))
    else:
        temp_below.append(np.nan)
plt.plot(ir_ts, temp_below, c='red', label='temperature below')
plt.title('Average Temperature Beneath')
plt.ylabel('Temperature (C)')
plt.xlabel('Time (s)')
plt.legend()
plt.show()
