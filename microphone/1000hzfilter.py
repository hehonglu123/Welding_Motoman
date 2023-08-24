import numpy as np
import scipy.signal
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import butter, lfilter

file_path = '../data/wall_weld_test/weld_scan_2023_08_23_15_23_45/layer_4/'
# 加载音频文件
y, sr = librosa.load(file_path + "mic_recording.wav", sr=None)

# # 设计一个低通滤波器
# nyquist = 0.5 * sr
# cutoff = 1000  # Desired cutoff frequency, in Hz
# normal_cutoff = cutoff / nyquist
# b, a = scipy.signal.butter(6, normal_cutoff, btype='low', analog=False)
#
# # 应用滤波器
# y_filtered = scipy.signal.filtfilt(b, a, y)
#
# # FFT验证
# D_original = np.abs(librosa.stft(y))
# D_filtered = np.abs(librosa.stft(y_filtered))


# 设计高通滤波器
def highpass_filter(data, sr, cutoff=1000):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    y_highpassed = lfilter(b, a, data)
    return y_highpassed

# 过滤掉低于1000Hz的信号
y_highpassed = highpass_filter(y, sr)
# Plot
plt.figure(figsize=(12, 6))

# Original Signal
plt.subplot(2, 1, 1)
plt.plot(y, color='blue')
plt.title('Original Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

# High-pass Filtered Signal
plt.subplot(2, 1, 2)
plt.plot(y_highpassed, color='red')
plt.title('High-pass Filtered Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# 计算STFT
D_original = librosa.stft(y)
D_filtered = librosa.stft(y_highpassed)

# Plot
plt.figure(figsize=(10, 4))

# Original Spectrum
plt.subplot(1, 2, 1)
librosa.display.specshow(librosa.amplitude_to_db(D_original, ref=np.max),
                         y_axis='log', x_axis='time')
plt.title('Original Spectrum')
# plt.colorbar()

# Filtered Spectrum
plt.subplot(1, 2, 2)
librosa.display.specshow(librosa.amplitude_to_db(D_filtered, ref=np.max),
                         y_axis='log', x_axis='time')
plt.title('Filtered Spectrum')
# plt.colorbar()

plt.tight_layout()
plt.show()
# 保存处理后的音频
sf.write(file_path + "mic_recording_filter.wav", y_highpassed, sr)


# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# librosa.display.specshow(librosa.amplitude_to_db(D_original, ref=np.max),
#                          y_axis='log', x_axis='time')
# plt.title('Original Spectrum')
# plt.colorbar()
#
# plt.subplot(1, 2, 2)
# librosa.display.specshow(librosa.amplitude_to_db(D_filtered, ref=np.max),
#                          y_axis='log', x_axis='time')
# plt.title('Filtered Spectrum')
# plt.colorbar()
#
# plt.tight_layout()
# plt.show()

# # 保存滤波后的音频到新的文件
# output_path = "../data/wall_weld_test/moveL_100_repeat_weld_scan_2023_08_02_17_07_02/layer_0/mic_recording_filter.wav"
# sf.write(output_path, y_filtered, sr)