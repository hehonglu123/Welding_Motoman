import numpy as np
import matplotlib.pyplot as plt
import pickle, sys
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from tkinter import *
import time
import os
sys.path.append('../toolbox/')
from flir_toolbox import *
freq=13
vmin_value = 0
vmax_value = 1300
ir_recordings = []
counts_all_frames = []
temp_all_frames = []
all_frames = []
data_mode = 0
main_folder_path = '../data/wall_weld_test/ER4043_correction_100ipm_2023_09_27_20_53_05'
for folder_name in os.listdir(main_folder_path):
    if folder_name.startswith('layer_'):
        folder_path = os.path.join(main_folder_path, folder_name)
        file_path = os.path.join(folder_path, 'ir_recording.pickle')
        
        file_size_kb = os.path.getsize(file_path) / 1024
        if file_size_kb > 1000:
            if os.path.isfile(file_path):
                with open(file_path, 'rb') as file:
                    ir_recording = pickle.load(file)
                    for i in range(len(ir_recording)):
                        temp = counts2temp(ir_recording[i].flatten(),6.39661118e+03, 1.40469989e+03, 1.00000008e+00, 8.69393436e+00, 8.40029488e+03,Emiss=0.13).reshape((240,320))
                        temp[temp > 1300] = 1300
                        temp_all_frames.append(temp)
                        # print(np.max(temp))
                    counts_all_frames.extend(ir_recording)

global interval
global frame_index
global direction
interval = 0.05
frame_index = 0
direction = 1

def update_animation_data():
    global all_frames, data_mode, vmin_slider, vmax_slider, vmin_value, vmax_value, cbar
    if data_mode == 0:
        all_frames = temp_all_frames
        vmin_slider.config(from_=0, to=2000)
        vmax_slider.config(from_=0, to=3500)
        vmin_slider.set(0)
        vmax_slider.set(1300)
        im.set_clim(0, 1300)  # 设置图像对象的上下限
        cbar.mappable.set_clim(0, 1300) # 这里设置colorbar范围
    else:
        all_frames = counts_all_frames
        vmin_slider.config(from_=0, to=8000)
        vmax_slider.config(from_=8000, to=20000)
        vmin_slider.set(8000)
        vmax_slider.set(12000)
        im.set_clim(0, 12000)  # 设置图像对象的上下限
        cbar.mappable.set_clim(0, 12000) # 这里设置colorbar范围

    data_mode = 1 - data_mode

    if all_frames:
        ani.event_source.stop()
        im.set_data(all_frames[0])
        im.set_clim(vmin=vmin_value, vmax=vmax_value)
        frame_index = 0
        ani.event_source.start()

all_frames = temp_all_frames

def update(frame):
    global frame_index
    im.set_array(all_frames[frame_index])
    im.set_clim(vmin=vmin_value, vmax=vmax_value)
    time.sleep(interval)
    frame_index = (frame_index + direction) % len(all_frames)
    return im,

fig, ax = plt.subplots()
plt.title('Test')
im = ax.imshow(all_frames[0], animated=True, cmap="inferno", aspect='auto')
im.set_clim(vmin=vmin_value, vmax=vmax_value)
cbar = plt.colorbar(im, format='%.2f')
cbar.set_label('Temperature (C)',rotation = 270, labelpad = 15)
ani = animation.FuncAnimation(fig, update, frames=len(all_frames), interval=1, blit=True)



root = Tk()
frame = Frame(root)
frame.pack()

def play():
    global direction
    direction = 1
    ani.event_source.start()

def reverse():
    global direction
    direction = -1
    ani.event_source.start()

def pause():
    ani.event_source.stop()

def set_speed(val):
    global interval
    interval = 1/(30.*float(val))

def set_vmin(val):
    global vmin_value
    vmin_value = float(val)

def set_vmax(val):
    global vmax_value
    vmax_value = float(val)

play_button = Button(frame, text="Play", command=play)
play_button.pack(side=LEFT)
reverse_button = Button(frame, text="Reverse", command=reverse)
reverse_button.pack(side=LEFT)
pause_button = Button(frame, text="Pause", command=pause)
pause_button.pack(side=LEFT)

speed_slider = Scale(frame, from_=1, to=10, orient=HORIZONTAL, label="Speed", command=set_speed)
speed_slider.set(1000)
speed_slider.pack(side=LEFT)

vmin_slider = Scale(frame, from_=0, to=8000, orient=VERTICAL, label="Lower Limit", command=set_vmin)
vmin_slider.pack(side=LEFT)
vmax_slider = Scale(frame, from_=8000, to=20000, orient=VERTICAL, label="Upper Limit", command=set_vmax)
vmax_slider.pack(side=LEFT)

toggle_button = Button(frame, text="Counts/Temperature mode", command=update_animation_data)
toggle_button.pack(side=LEFT)

plt.show(block=False)
root.mainloop()
