import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('2023_04_27_17_38_23ir_recording.pickle', 'rb') as file:
    ir_recording=pickle.load(file)
freq=13

fig = plt.figure(1)
for i in range(len(ir_recording)):
    plt.imshow(ir_recording[0], cmap='inferno', aspect='auto')
    plt.colorbar(format='%.2f')
plt.pause(1/freq)
plt.clf()
