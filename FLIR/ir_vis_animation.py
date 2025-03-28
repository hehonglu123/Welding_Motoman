import numpy as np
import matplotlib.pyplot as plt
import pickle, sys
from matplotlib.animation import FuncAnimation

data_dir='../data/wall_weld_test/weld_fujiscan_2025_02_26_18_08_18/layer264/'
# data_dir='../data/wall_weld_test/weld_fujicontrol_2025_03_12_16_14_17/layer203/'
with open(data_dir+'ir_recording.pickle', 'rb') as file:
    ir_recording=pickle.load(file)

total_img = len(ir_recording)
show_id = int(total_img/2)

fig, ax = plt.subplots()
im = ax.imshow(np.rot90(ir_recording[0],k=3), cmap='inferno', aspect='auto')

def update(frame):
    im.set_array(np.rot90(ir_recording[frame],k=3))
    return [im]

ani = FuncAnimation(fig, update, frames=range(total_img), interval=10, blit=True)
plt.show()