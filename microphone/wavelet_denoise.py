import numpy as np
import pywt
import numpy as np
import wave, copy
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import os

def wavelet_denoise(data, wavelet='db1', level=4, threshold_type='soft'):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    coeffs_thresholded = [pywt.threshold(c, threshold, mode=threshold_type) for c in coeffs]
    return pywt.waverec(coeffs_thresholded, wavelet)
# 计算音频的频谱
def plot_spectrum(samples, fs):
    N = len(samples)
    yf = np.fft.fft(samples)
    xf = np.fft.fftfreq(N, 1/fs)
    plt.plot(xf[:N//2], 2.0/N * np.abs(yf[:N//2]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()

# 加载音频文件
y, sr = librosa.load("../data/wall_weld_test/moveL_100_repeat_weld_scan_2023_08_02_17_07_02/layer_10/mic_recording.wav", sr=None)

# 剪切前3秒
start_sample = int(6 * sr)
y = y[start_sample:]

# 剪切最后1秒
end_sample = -1 * sr
y = y[:end_sample]

# 保存处理后的音频
filepath = "../data/wall_weld_test/moveL_100_repeat_weld_scan_2023_08_02_17_07_02/layer_10/mic_recording_cut.wav"
y, sr = sf.read(filepath)
# 获取文件所在的路径
audio_directory = os.path.dirname(filepath)

with wave.open("../data/wall_weld_test/moveL_100_repeat_weld_scan_2023_08_02_17_07_02/layer_10/mic_recording_cut.wav", "rb") as wf:
    n_samples = wf.getnframes()
    audio_data = wf.readframes(n_samples)
    audio_samples = np.frombuffer(audio_data, dtype=np.int16)

fs = 44000  # 采样率
t = np.arange(len(audio_samples)) / fs  # 时间轴

# Example usage:
# (assuming audio_samples is a numpy array containing your audio data)
denoised_samples = wavelet_denoise(audio_samples)

# 根据denoised_samples重新计算时间轴
t_denoised = np.arange(len(denoised_samples)) / 44000

# 绘制原始信号和降噪信号
t_original = np.arange(len(audio_samples)) / 44000  # 原始信号的时间轴

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t_original, audio_samples, color='blue', label="Original Signal")
plt.title("Original Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_denoised, denoised_samples, color='red', label="Denoised Signal")
plt.title("Denoised Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plot_spectrum(audio_samples, 44000)
plt.title("Original Signal Spectrum")

plt.subplot(2, 1, 2)
plot_spectrum(denoised_samples, 44000)
plt.title("Denoised Signal Spectrum")

plt.tight_layout()
plt.show()

# 在音频文件所在的路径下创建新文件夹
output_folder = os.path.join(audio_directory, "microphone_segments")
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# 计算每个片段的长度
segment_length = len(audio_samples) // 20

# 分割音频并保存
for i in range(20):
    segment = audio_samples[i*segment_length: (i+1)*segment_length]
    sf.write(os.path.join(output_folder, f'segments_{i}.wav'), segment, fs)