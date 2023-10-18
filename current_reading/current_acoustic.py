import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import resample

base_path = '../data/wall_weld_test/ER4043_correction_100ipm_2023_09_27_20_53_05/layer_7/'
fs_wav, audio_data = wav.read(base_path + "mic_recording.wav")


df = pd.read_csv(base_path + "current.csv")
time_stamps = df['timestamp'].values
current_signal = df['current'].values


plt.figure(figsize=(10, 6))

# 绘制声音信号
plt.subplot(2, 1, 1)
audio_duration = len(audio_data) / fs_wav  # 计算音频的持续时间
plt.plot(np.linspace(0, audio_duration, len(audio_data)), audio_data)
plt.title('Audio Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# 绘制current信号
plt.subplot(2, 1, 2)
plt.plot(time_stamps, current_signal)
plt.title('Current Signal')
plt.xlabel('Time (s)')
plt.ylabel('Current Value')

plt.tight_layout()
plt.show()

t_start = 4.3  # 起始时间，你可以根据需要修改
t_end = t_start + 0.2  # 结束时间

# 从音频信号中截取
audio_start_idx = int(t_start * fs_wav)
audio_end_idx = int(t_end * fs_wav)
audio_subset = audio_data[audio_start_idx:audio_end_idx]

# 从电流信号中截取
current_subset = df[(df['timestamp'] >= t_start) & (df['timestamp'] <= t_end)]['current'].values
time_subset = df[(df['timestamp'] >= t_start) & (df['timestamp'] <= t_end)]['timestamp'].values

# 绘制
plt.figure(figsize=(10, 6))

# 绘制截取的音频信号
plt.subplot(2, 1, 1)
plt.plot(np.linspace(t_start, t_end, len(audio_subset)), audio_subset)
plt.title('Audio Signal Subset')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# 绘制截取的电流信号
plt.subplot(2, 1, 2)
plt.plot(time_subset, current_subset)
plt.title('Current Signal Subset')
plt.xlabel('Time (s)')
plt.ylabel('Current Value')

plt.tight_layout()
plt.show()