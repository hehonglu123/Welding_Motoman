import numpy as np
import matplotlib.pyplot as plt
import pickle, sys
import matplotlib.animation as animation
from tkinter import *
import time
import os
import glob

with open('recorded_data/layer_150/ir_recording.pickle', 'rb') as file:
    ir_recording=pickle.load(file)
print(len(ir_recording))
freq=13
rot_ir_recording = [None] * len(ir_recording)
vmin = 8000
vmax = 10000
tmin = 500
tmax = 725
# fig = plt.figure(1)
for i in range(len(ir_recording)):
    print(np.max(ir_recording[i]),np.min(ir_recording[i]))
    rot_ir_recording[i] = np.rot90(ir_recording[i],-1)
    # plt.imshow(rot_ir_recording[i], cmap='inferno', aspect='auto')
    # plt.imshow(rot_ir_recording[i], cmap='inferno', aspect='auto',vmax=vmax,vmin=vmin)
    # plt.imshow(rot_ir_recording[i]*0.1- 273.15 , cmap='inferno', aspect='auto')
    # plt.imshow(rot_ir_recording[i]*0.1- 273.15 , cmap='inferno', aspect='auto',vmax=tmax)
    # plt.colorbar(format='%.2f')
    # plt.pause(1/freq)
    # plt.clf()

# 全局变量
global interval
global frame_index
global direction
interval = 0.05
frame_index = 0
direction = 1  # 1 表示向前播放，-1 表示回退

def update(frame):
    global frame_index
    im.set_array(rot_ir_recording[frame_index])
    time.sleep(interval)
    frame_index = (frame_index + direction) % len(rot_ir_recording)
    return im,

# 创建动画
fig, ax = plt.subplots()
im = ax.imshow(rot_ir_recording[0], animated=True, cmap="inferno", aspect='auto', vmax=vmax, vmin=vmin)
ani = animation.FuncAnimation(fig, update, frames=len(rot_ir_recording), interval=1, blit=True)

# 创建Tkinter窗口
root = Tk()
frame = Frame(root)
frame.pack()

# 播放控制函数
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

# 添加控制按钮
play_button = Button(frame, text="Play", command=play)
play_button.pack(side=LEFT)
reverse_button = Button(frame, text="Reverse", command=reverse)
reverse_button.pack(side=LEFT)
pause_button = Button(frame, text="Pause", command=pause)
pause_button.pack(side=LEFT)

# 添加滑块以控制速度
speed_slider = Scale(frame, from_=10, to=200, orient=HORIZONTAL, label="Speed", command=set_speed)
speed_slider.set(200) # 初始速度设为最快
speed_slider.pack(side=LEFT)

# 显示matplotlib图窗口
plt.show(block=False)

root.mainloop()