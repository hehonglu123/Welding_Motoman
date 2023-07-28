import cv2,wave,copy
import pickle, sys
import numpy as np
import scipy.signal as signal
sys.path.append('../toolbox/')
from flir_toolbox import *
import matplotlib.pyplot as plt

data_dir="../../recorded_data/weld_scan_job205_v152023_07_27_13_23_06/layer_150/"

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
# Load the IR recording data from the pickle file
with open(data_dir+'ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)
ir_ts=np.loadtxt(data_dir+'ir_stamps.csv', delimiter=',')
ir_ts=ir_ts-ir_ts[0]
center_brightness=np.average(ir_recording[:,100:140,140:180],axis=(1,2))
ax1.plot(ir_ts, center_brightness, label='ir center counts')
ax1.set_ylabel('Center Brightness (counts)')

##microphone data
wavfile = wave.open('../../recorded_data/weld_scan_job205_v152023_07_27_13_23_06/layer_150/mic_recording.wav', 'rb')
samplerate = 44000
channels = 1
audio_data=np.frombuffer(wavfile.readframes(wavfile.getnframes()),dtype=np.int16)
_, _, Sxx = signal.spectrogram(audio_data, samplerate)
mean_freq = np.sum(Sxx * np.arange(Sxx.shape[0])[:, np.newaxis], axis=0) / np.sum(Sxx, axis=0)
frame_time = np.arange(mean_freq.shape[0]) * len(audio_data) / Sxx.shape[1] / samplerate

ax2.plot(frame_time, mean_freq, label='microphone frequency')
ax2.set_xlabel('Time Frame')
ax2.set_ylabel('Average Frequency (Hz)')
ax2.set_title('Average Frequency vs Time')
# plt.show()

##welding data
welding_data=np.loadtxt(data_dir+'welding.csv',skiprows=1, delimiter=',')
welding_data[:,0]-=welding_data[0,0]
welding_data[:,0]/=1e6
ax2.plot(welding_data[:,0],welding_data[:,1],label='voltage')
ax2.plot(welding_data[:,0],welding_data[:,2],label='current')
ax2.plot(welding_data[:,0],welding_data[:,3],label='feedrate')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc=1)
plt.show()

##height profile
profile_height=np.load(data_dir+'scans/height_profile.npy')
plt.scatter(profile_height[:,0],profile_height[:,1])
plt.show()

