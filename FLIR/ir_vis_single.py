import numpy as np
import matplotlib.pyplot as plt
import pickle, sys

data_dir='../data/wall_weld_test/moveL_100_weld_scan_2023_07_24_11_19_58/layer_15/'
with open(data_dir+'ir_recording.pickle', 'rb') as file:
    ir_recording=pickle.load(file)

total_img = len(ir_recording)
show_id = int(total_img/2)

plt.imshow(ir_recording[show_id], cmap='inferno', aspect='auto')
plt.colorbar(format='%.2f')
plt.show()
