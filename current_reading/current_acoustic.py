import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import resample
from scipy.interpolate import interp1d
import librosa
import os

def load_height_profile(base_path, layer_num):
    path = os.path.join(base_path.replace('#', str(layer_num)), 'scans', 'height_profile.npy')
    if os.path.exists(path):
        return np.load(path)
    else:
        return None

def compute_dh(base_path, layer_num):
    current_profile = load_height_profile(base_path, layer_num)
    if current_profile is None:
        return None
    
    if layer_num == 0:
        return current_profile
    
    previous_profile = load_height_profile(base_path, layer_num - 1)
    if previous_profile is None:
        return None
    
    # if layer_num % 2 == 1:  # 当前层为反向
    #     previous_profile = previous_profile[::-1]
    
    # 如果两层数据长度不同，使它们具有相同长度
    min_length = min(len(current_profile), len(previous_profile))
    current_profile = current_profile[:min_length]
    previous_profile = previous_profile[:min_length]
    
    dh = current_profile - previous_profile
    return dh

base_path = '../data/wall_weld_test/ER4043_correction_100ipm_2023_09_27_20_53_05/layer_#/'
layer_num = 4  # 修改为你的层数
dh = compute_dh(base_path, layer_num)
tip_dis = (np.mean(dh) + 2.3420716473455623) - dh
base_path = '../data/wall_weld_test/ER4043_correction_100ipm_2023_09_27_20_53_05/layer_4/'
fs_wav, audio_data = wav.read(base_path + "mic_recording.wav")
height_path = '../data/wall_weld_test/ER4043_correction_100ipm_2023_09_27_20_53_05/'

df = pd.read_csv(base_path + "current.csv")
time_stamps = df['timestamp'].values
current_signal = df['current'].values

# 读取坐标文件
height_profile = np.load(base_path + 'scans/' + "height_profile.npy")
print('print(height_profile.shape):',height_profile.shape)
# 创建与电流信号相同长度的时间轴
time_axis = np.linspace(0, len(current_signal) / fs_wav, len(current_signal))
print('time_axis:',time_axis.shape)

# 创建一个新的图形
# 1. 计算电流信号的平均值
threshold_start = 10
threshold_end = 10
# 2. 寻找电流信号中第一个超过阈值的点
start_index = np.where(current_signal > threshold_start)[0][0]

# 3. 从起始点开始，寻找电流信号中第一个低于阈值的点
end_index = np.where(current_signal[start_index:] < threshold_end)[0][0] + start_index

# 4. 使用这两个时间点截取声音信号
audio_start_index = int(start_index * len(audio_data) / len(current_signal))
audio_end_index = int(end_index * len(audio_data) / len(current_signal))
audio_clipped = audio_data[audio_start_index:audio_end_index]

# 对height_profile进行线性插值，使其与time_axis长度相同
height_interpolated = interp1d(np.linspace(0, 1, len(dh)), dh[:, 1])(np.linspace(0, 1, end_index - start_index))
tip_interpolated = interp1d(np.linspace(0, 1, len(tip_dis)), tip_dis[:, 1])(np.linspace(0, 1, end_index - start_index))
# 先解析base_path中的layer_#部分
layer_number = int(base_path.split('layer_')[-1].split('/')[0])

# 判断layer_number是奇数还是偶数
if layer_number % 2 == 0:  # 偶数
    direction = "left-to-right"
else:  # 奇数
    direction = "right-to-left"

# # 如果是从右往左，就反转height_profile
if direction == "left-to-right":
    height_interpolated = height_interpolated[::-1]
    tip_interpolated = tip_interpolated[::-1]

# 以下是展示截取后的数据
plt.figure(figsize=(10, 8))

# 确定电流信号的时间轴的起始和结束时间
current_time_start = time_stamps[start_index]
current_time_end = time_stamps[end_index]

# 计算截取的声音信号的时间轴
audio_time_stamps = np.linspace(current_time_start, current_time_end, len(audio_clipped))

# 绘制截取后的声音信号
plt.subplot(3, 1, 1)
plt.plot(audio_time_stamps, audio_clipped)
plt.title('Acoustic Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (Pa)')

# 绘制截取后的电流信号
plt.subplot(3, 1, 2)
plt.plot(time_stamps[start_index:end_index], current_signal[start_index:end_index])
plt.title('Current Signal')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')

# 绘制截取后的坐标信号
plt.subplot(3, 1, 3)
plt.plot(time_stamps[start_index:end_index], height_interpolated)
if direction == "left-to-right":
    plt.title('Layer by Layer Height Profile')
else:
    plt.title('Layer by Layer Height Profile')
plt.xlabel('Time (s)')
plt.ylabel('Height (mm)')

# 绘制截取后的坐标信号
# plt.subplot(4, 1, 4)
# plt.plot(time_stamps[start_index:end_index], tip_interpolated)
# if direction == "left-to-right":
#     plt.title('Tip Distance Profile (Left to Right)')
# else:
#     plt.title('Tip Distance Profile (Right to Left)')
# plt.xlabel('Time (s)')
# plt.ylabel('Tip Distance (mm)')
plt.tight_layout()
plt.show()
###################################################################################################################
t_start = 4.3  # 起始时间，你可以根据需要修改
t_end = t_start + 0.3  # 结束时间

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
########################################################################################################################
# 1. Calculate energy of audio signal with respect to time

frame_length = int(fs_wav * 0.02)  # 20 ms frames
hop_length = int(fs_wav * 0.01)  # 10 ms overlap
audio_clipped = audio_clipped.astype(np.float32) / np.iinfo(np.int16).max

# Calculate energy for each frame
energy = np.array([np.sum(np.abs(audio_clipped[i:i+frame_length]**2)) for i in range(0, len(audio_clipped), hop_length)])

# 2. Calculate RMSE with respect to time
rmse_time = librosa.feature.rms(y=audio_clipped, frame_length=frame_length, hop_length=hop_length)

# 3. Calculate RMSE with respect to frequency
D = librosa.stft(audio_clipped, n_fft=2048, hop_length=hop_length, win_length=frame_length)

spectral_rms = np.sqrt(np.mean(np.abs(D)**2, axis=1))

# Plotting
plt.figure(figsize=(15, 10))

# Energy vs Time
plt.subplot(3, 1, 1)
plt.plot(np.arange(len(energy)) * hop_length / fs_wav, energy)
plt.title('Energy vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Energy')

# RMSE vs Time
plt.subplot(3, 1, 2)
times = np.arange(rmse_time.shape[1]) * hop_length / fs_wav
plt.plot(times, rmse_time[0])
plt.title('RMSE vs Time')
plt.xlabel('Time (s)')
plt.ylabel('RMSE')

# RMSE vs Frequency
plt.subplot(3, 1, 3)
freqs = librosa.fft_frequencies(sr=fs_wav, n_fft=2048)
plt.plot(freqs, spectral_rms)
plt.title('RMSE vs Frequency')
plt.xlabel('Frequency (Hz)')
plt.ylabel('RMSE')
plt.xlim([0, fs_wav / 2])  # display only up to Nyquist frequency

plt.tight_layout()
plt.show()