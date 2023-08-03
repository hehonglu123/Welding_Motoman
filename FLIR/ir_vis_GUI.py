import numpy as np
import matplotlib.pyplot as plt
import pickle, sys
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from tkinter import *
import time
import os
import glob

freq=13
vmin = 8000
vmax = 10000
vmin_value = 0
vmax_value = 10000
tmin = 500
tmax = 725
# 初始化一个空列表来保存所有的ir_recording对象
ir_recordings = []
all_frames = []
# 你的主文件夹路径
main_folder_path = '../data/wall_weld_test/moveL_100_weld_scan_2023_08_02_15_17_25'

for folder_name in os.listdir(main_folder_path):
    if folder_name.startswith('layer_'):
        # 构建完整的文件夹路径和文件路径
        folder_path = os.path.join(main_folder_path, folder_name)
        file_path = os.path.join(folder_path, 'ir_recording.pickle')

        # 检查文件大小是否大于1000KB
        file_size_kb = os.path.getsize(file_path) / 1024
        if file_size_kb > 1000:
            # 如果文件存在并且大于1000KB，则读取并附加到列表中
            if os.path.isfile(file_path):
                with open(file_path, 'rb') as file:
                    ir_recording = pickle.load(file)
                    all_frames.extend(ir_recording)  # 添加到总列表

# 全局变量
global interval
global frame_index
global direction
interval = 0.05
frame_index = 0
direction = 1  # 1 表示向前播放，-1 表示回退

def update(frame):
    global frame_index
    im.set_array(all_frames[frame_index])
    im.set_clim(vmin=vmin_value, vmax=vmax_value)  # 更新 vmin 和 vmax
    time.sleep(interval)
    frame_index = (frame_index + direction) % len(all_frames)
    return im,

# 创建动画
fig, ax = plt.subplots()
plt.title(main_folder_path)
im = ax.imshow(all_frames[0], animated=True, cmap="inferno", aspect='auto')
im.set_clim(vmin=vmin_value, vmax=vmax_value)  # 初始化 vmin 和 vmax
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
def set_vmin(val):
    global vmin_value
    vmin_value = float(val)

def set_vmax(val):
    global vmax_value
    vmax_value = float(val)


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
# 控制 vmin 的滑块
vmin_slider = Scale(frame, from_=0, to=8000, orient=VERTICAL, label="VMin", command=set_vmin)
vmin_slider.pack(side=LEFT)

# 控制 vmax 的滑块
vmax_slider = Scale(frame, from_=8000, to=20000, orient=VERTICAL, label="VMax", command=set_vmax)
vmax_slider.pack(side=LEFT)
vmin_slider = Scale(frame, from_=0, to=8000, orient=VERTICAL, label="VMin", length=200, command=set_vmin)
vmax_slider = Scale(frame, from_=8000, to=20000, orient=VERTICAL, label="VMax", length=200, command=set_vmax)

# 显示matplotlib图窗口
plt.show(block=False)
root.mainloop()