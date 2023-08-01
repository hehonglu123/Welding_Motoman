import numpy as np
import matplotlib.pyplot as plt
import pickle, sys

with open('recorded_data/slice_180_0_flir.pickle', 'rb') as file:
    ir_recording=pickle.load(file)
freq=13

fig = plt.figure(1)
for i in range(len(ir_recording)):
    print(np.max(ir_recording[i]),np.min(ir_recording[i]))
    plt.imshow(ir_recording[i], cmap='inferno', aspect='auto')
    # plt.imshow(ir_recording[i]*0.1- 273.15 , cmap='inferno', aspect='auto')
    plt.colorbar(format='%.2f')
    plt.pause(1/freq)
    plt.clf()
