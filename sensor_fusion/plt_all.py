import cv2,wave,copy
import pickle, sys
import numpy as np
import scipy.signal as signal
sys.path.append('../toolbox/')
from flir_toolbox import *
import matplotlib.pyplot as plt

data_dir="../data/wall_weld_test/moveL_100_repeat_weld_scan_2023_08_02_17_07_02/layer_10/"

fig, ax1 = plt.subplots()
ax1.set_title('Welding All Data')
ax2 = ax1.twinx()
ax3 = ax2.twiny()
# Load the IR recording data from the pickle file
with open(data_dir+'ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)
ir_ts=np.loadtxt(data_dir+'ir_stamps.csv', delimiter=',')
ir_ts=ir_ts-ir_ts[0]
center_brightness=np.average(ir_recording[:,100:140,140:180],axis=(1,2))
ax1.plot(ir_ts, center_brightness, c='red', label='ir center counts')
ax1.set_ylabel('Center Brightness (counts)')
ax1.set_xlabel('Time (s)')

##microphone data
wavfile = wave.open('../data/wall_weld_test/moveL_100_repeat_weld_scan_2023_08_02_17_07_02/layer_10/mic_recording.wav', 'rb')
samplerate = 44000
channels = 1
audio_data=np.frombuffer(wavfile.readframes(wavfile.getnframes()),dtype=np.int16)
freq, mic_ts, Sxx = signal.spectrogram(audio_data, samplerate)
mean_freq = np.sum(Sxx * np.arange(Sxx.shape[0])[:, np.newaxis], axis=0) / np.sum(Sxx, axis=0)
s = np.flatnonzero(mean_freq > 20)
start_welding_mic,end_welding_mic=mic_ts[s[0]], mic_ts[s[-1]]
print(start_welding_mic,end_welding_mic)
welding_duration_mic=end_welding_mic-start_welding_mic
welding_duration_mic_ext=mic_ts[-1]-mic_ts[0]
print('Welding Duration: ',welding_duration_mic)


ax2.plot(mic_ts, mean_freq, label='microphone frequency')
ax2.set_xlabel('Time Frame')
ax2.set_ylabel('Frequency, Height, Current, Voltage, Feedrate')
# plt.show()

##welding data
welding_data=np.loadtxt(data_dir+'welding.csv',skiprows=1, delimiter=',')
welding_data[:,0]-=welding_data[0,0]
welding_data[:,0]/=1e6
ax2.plot(welding_data[:,0],welding_data[:,1],label='voltage')
ax2.plot(welding_data[:,0],welding_data[:,2],label='current')
ax2.plot(welding_data[:,0],welding_data[:,3],c='gray',label='feedrate')


# plt.show() 

##height profile
profile_height=np.load(data_dir+'scans/height_profile.npy')
welding_length=profile_height[-1,0]-profile_height[0,0]

welding_length_ext=welding_length*welding_duration_mic_ext/welding_duration_mic
welding_ext_start=profile_height[0,0]-welding_length_ext*start_welding_mic/welding_duration_mic_ext

ax3.set_xlim(welding_ext_start,welding_ext_start+welding_length_ext)
ax1.set_xlim(mic_ts[0],mic_ts[-1])
ax3.scatter(profile_height[:,0],profile_height[:,1], c='silver', label='height profile')
ax3.set_xlabel('Distance (mm)')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
h3, l3 = ax3.get_legend_handles_labels()
ax1.legend(h1+h2+h3, l1+l2+l3, loc=1)
plt.show()

profile_height = np.load(data_dir + 'scans/height_profile.npy')

# 计算每段的长度
# num_segments = 20
# segment_length = len(profile_height) // num_segments

# color_cycle = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999"]
# colors = [color_cycle[i % len(color_cycle)] for i in range(num_segments)]

# plt.figure(figsize=(10, 6))

# # 循环遍历每段并绘制
# for i in range(num_segments):
#     start_index = i * segment_length
#     end_index = (i + 1) * segment_length if i != num_segments - 1 else None  # 如果是最后一段，取到末尾
    
#     segment_data = profile_height[start_index:end_index]
#     plt.plot(segment_data[:, 0], segment_data[:, 1], color=colors[i], label='Height Profile' if i == 0 else "")

# # 如果您只想为一个段加上legend，您可以在上述循环中只为一个段添加label。如上例中，只为第一个segment设置了label。

# plt.xlabel('Distance (mm)')
# plt.ylabel('Height')
# plt.title('Height Profile with Segments')
# plt.legend()
# plt.show()