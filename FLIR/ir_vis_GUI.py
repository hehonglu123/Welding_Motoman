import numpy as np
import matplotlib.pyplot as plt
import pickle, sys
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from tkinter import *
import time
import os
sys.path.append('../toolbox/')
from flir_toolbox import *

freq=13
vmin_value = 0
vmax_value = 1300
# Empty list creation
ir_recordings = []
counts_all_frames = []
temp_all_frames=[]
all_frames = []
# Initialize data mode (0 for ir_recording, 1 for temperature)
data_mode = 0
# Local folder path
main_folder_path = '../data/wall_weld_test/moveL_100_repeat_weld_scan_2023_08_02_17_07_02'

for folder_name in os.listdir(main_folder_path):
    if folder_name.startswith('layer_'):
        folder_path = os.path.join(main_folder_path, folder_name)
        file_path = os.path.join(folder_path, 'ir_recording.pickle')

        # Filter less than 1000KB
        file_size_kb = os.path.getsize(file_path) / 1024
        if file_size_kb > 1000:
            if os.path.isfile(file_path):
                with open(file_path, 'rb') as file:
                    ir_recording = pickle.load(file)
                    for i in range(len(ir_recording)):
                        temp = counts2temp(ir_recording[i].flatten(),6.39661118e+03, 1.40469989e+03, 1.00000008e+00, 8.69393436e+00, 8.40029488e+03,Emiss=0.13).reshape((240,320))
                        temp_all_frames.append(temp.astype(np.uint8))
                        print(np.max(temp))
                    counts_all_frames.extend(ir_recording)

# Global parameters
global interval
global frame_index
global direction
interval = 0.05
frame_index = 0
direction = 1  # 1 play，-1 reverse
def update_animation_data():
    global all_frames, data_mode, vmin_slider, vmax_slider, vmin_value, vmax_value
    if data_mode == 0:
        all_frames = temp_all_frames
        vmin_slider.config(from_=0, to=2000)  # Update vmin_slider range for temperature mode
        vmax_slider.config(from_=0, to=2000)  # Update vmax_slider range for temperature mode
        vmin_slider.set(vmin_value)  # Set vmin_slider value for temperature mode
        vmax_slider.set(vmax_value)  # Set vmax_slider value for temperature mode
    else:
        all_frames = counts_all_frames
        vmin_slider.config(from_=0, to=8000)  # Update vmin_slider range for counts mode
        vmax_slider.config(from_=8000, to=20000)  # Update vmax_slider range for counts mode
        vmin_slider.set(vmin_value)  # Set vmin_slider value for counts mode
        vmax_slider.set(vmax_value)  # Set vmax_slider value for counts mode

    data_mode = 1 - data_mode  # Toggle data mode

    # Ensure all_frames is not empty before updating the animation
    if all_frames:
        ani.event_source.stop()  # Stop the current animation
        im.set_data(all_frames[0])  # Update the image data
        im.set_clim(vmin=vmin_value, vmax=vmax_value)  # Update vmin and vmax
        frame_index = 0
        ani.event_source.start()  # Start the animation

# Set initial animation data to temp_all_frames
all_frames = temp_all_frames
def update(frame):
    global frame_index
    im.set_array(all_frames[frame_index])
    im.set_clim(vmin=vmin_value, vmax=vmax_value)  
    time.sleep(interval)
    frame_index = (frame_index + direction) % len(all_frames)
    return im,

# Create separate figure for the colorbar
fig_cbar, ax_cbar = plt.subplots(figsize=(1, 4))
fig_cbar.subplots_adjust(left=0.1, right=0.5)
norm = Normalize(vmin=vmin_value, vmax=vmax_value)
cbar = ColorbarBase(ax_cbar, cmap="inferno", orientation="vertical", norm=norm)
cbar.set_label('Temperature', rotation=270, labelpad=15)

def update_colorbar():
    norm.vmin = vmin_value
    norm.vmax = vmax_value
    cbar.update_normal(cbar.mappable)
    fig_cbar.canvas.draw()
# Animation creation
fig, ax = plt.subplots()
plt.title(main_folder_path)
im = ax.imshow(all_frames[0], animated=True, cmap="inferno", aspect='auto')
im.set_clim(vmin=vmin_value, vmax=vmax_value)
ani = animation.FuncAnimation(fig, update, frames=len(all_frames), interval=1, blit=True)

# fig = plt.figure(1)
# for i in range(len(ir_recording)):
#     # print(np.max(ir_recordings[i]),np.min(ir_recordings[i]))
#     rot_ir_recording[i] = np.rot90(ir_recordings[i],-1)
#     # plt.imshow(rot_ir_recording[i], cmap='inferno', aspect='auto')
#     # plt.imshow(rot_ir_recording[i], cmap='inferno', aspect='auto',vmax=vmax,vmin=vmin)
#     # plt.imshow(rot_ir_recording[i]*0.1- 273.15 , cmap='inferno', aspect='auto')
#     # plt.imshow(rot_ir_recording[i]*0.1- 273.15 , cmap='inferno', aspect='auto',vmax=tmax)
#     # plt.colorbar(format='%.2f')
#     # plt.pause(1/freq)
#     # plt.clf()

#  Create Tkinter Windows
root = Tk()
frame = Frame(root)
frame.pack()

# Control function
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
    interval = (200 - float(val)) / 1000
def set_vmin(val):
    global vmin_value
    vmin_value = float(val)

def set_vmax(val):
    global vmax_value
    vmax_value = float(val)


# Add control button
play_button = Button(frame, text="Play", command=play)
play_button.pack(side=LEFT)
reverse_button = Button(frame, text="Reverse", command=reverse)
reverse_button.pack(side=LEFT)
pause_button = Button(frame, text="Pause", command=pause)
pause_button.pack(side=LEFT)

# Add button to change speed
speed_slider = Scale(frame, from_=10, to=200, orient=HORIZONTAL, label="Speed", command=set_speed)
speed_slider.set(200)
speed_slider.pack(side=LEFT)

vmin_slider = Scale(frame, from_=0, to=8000, orient=VERTICAL, label="VMin", command=set_vmin)
vmin_slider.pack(side=LEFT)
vmax_slider = Scale(frame, from_=8000, to=20000, orient=VERTICAL, label="VMax", command=set_vmax)
vmax_slider.pack(side=LEFT)

# # Connect slider updates to the function
# vmin_slider.config(command=lambda val: [set_vmin(val), update_colorbar_limits(val)])
# vmax_slider.config(command=lambda val: [set_vmax(val), update_colorbar_limits(val)])

# Button to toggle between original and temperature data
toggle_button = Button(frame, text="Counts/Temperature mode", command=update_animation_data)
toggle_button.pack(side=LEFT)
# Show matplotlib window
plt.show(block=False)
root.mainloop()