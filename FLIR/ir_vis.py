import numpy as np
import matplotlib.pyplot as plt
import pickle, sys
sys.path.append('../toolbox/')
from flir_toolbox import *

with open('../data/wall_weld_test/weld_scan_100ipm_cool_2023_08_10_11_28_20/layer_1/ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)
freq=13

fig = plt.figure(1)
for i in range(len(ir_recording)):
    temp=counts2temp(ir_recording[i].flatten(),6.39661118e+03, 1.40469989e+03, 1.00000008e+00, 8.69393436e+00, 8.40029488e+03,Emiss=0.13).reshape((240,320))
    print(np.max(temp),np.min(temp))
    # temp_hdr=np.log(1 + temp) / np.log(1000)
    temp[temp > 1300] = 1300
    plt.imshow(temp, cmap='inferno', aspect='auto')
    plt.colorbar(format='%.2f')
    plt.pause(0.001)
    plt.clf()
