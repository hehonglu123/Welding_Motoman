import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

base_path = '../data/wall_weld_test/ER4043_correction_100ipm_2023_09_27_20_53_05/layer_3/'
# 假设您的数据是numpy数组
audio_clipped = np.load(base_path + 'audio_clipped.npy')
clipped_current = np.load(base_path + 'current_clipped.npy')
height_interpolated = np.load(base_path + 'height_interpolated.npy')

def segment_analysis(data, n_segments=40):
    segment_len = len(data) // n_segments
    means = []
    stds = []
    for i in range(n_segments):
        segment = data[i*segment_len:(i+1)*segment_len]
        means.append(np.mean(segment))
        stds.append(np.std(segment))
    return means, stds

def spectral_analysis(data, n_segments=40):
    segment_len = len(data) // n_segments
    spectra = []
    for i in range(n_segments):
        segment = data[i*segment_len:(i+1)*segment_len]
        spectrum = np.abs(fft(segment))
        spectra.append(spectrum)
    return spectra

# 对三组数据进行segment分析
audio_means, audio_stds = segment_analysis(audio_clipped)
current_means, current_stds = segment_analysis(clipped_current)
height_means, height_stds = segment_analysis(height_interpolated)

# 对audio_clipped和clipped_current进行频谱分析
audio_spectra = spectral_analysis(audio_clipped)
current_spectra = spectral_analysis(clipped_current)


# 使用之前的函数进行频谱分析
audio_spectra = spectral_analysis(audio_clipped)
current_spectra = spectral_analysis(clipped_current)

# 绘制结果
plt.figure(figsize=(12, 8))

# 绘制audio_clipped的mean和std
plt.subplot(3, 1, 1)
plt.plot(audio_means, label='Mean')
plt.plot(audio_stds, label='Std')
plt.title('audio_clipped Mean & Std')
plt.legend()

# 绘制clipped_current的mean和std
plt.subplot(3, 1, 2)
plt.plot(current_means, label='Mean')
plt.plot(current_stds, label='Std')
plt.title('clipped_current Mean & Std')
plt.legend()

# 绘制height_interpolated的mean和std
plt.subplot(3, 1, 3)
plt.plot(height_means, label='Mean')
plt.plot(height_stds, label='Std')
plt.title('height_interpolated Mean & Std')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
# 绘制audio_clipped的频谱分析
plt.subplot(2, 1, 1)
for segment_spectra in audio_spectra:
    plt.plot(segment_spectra)
plt.title('audio_clipped Spectra')

# 绘制clipped_current的频谱分析
plt.subplot(2, 1, 2)
for segment_spectra in current_spectra:
    plt.plot(segment_spectra)
plt.title('clipped_current Spectra')

plt.tight_layout()
plt.show()