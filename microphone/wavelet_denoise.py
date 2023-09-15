import numpy as np
import pywt
import numpy as np
import wave, copy
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import os
import re

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

base_path = '../data/wall_weld_test/moveL_100_baseline_weld_scan_2023_07_07_15_20_56/'

if os.path.exists(base_path):
    # 获取指定路径下的所有子目录
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    # 使用正则表达式匹配 layer_n 模式的子目录
    layer_dirs = [d for d in subdirs if re.match(r'layer_\d+', d)]

    for layer_dir in sorted(layer_dirs, key=lambda x: int(x.split('_')[-1])):
        layer_path = os.path.join(base_path, layer_dir + '/',)
        # Construct the path to the mic_recording.wav file
        mic_recording_path = os.path.join(layer_path, "mic_recording_filter.wav")
        
        # Check if mic_recording.wav exists in the current subdir
        if not os.path.exists(mic_recording_path):
            print(f"mic_recording_filter.wav not found in {layer_path}. Skipping...")
            continue  # Skip to the next iteration
        # file_path = '../data/wall_weld_test/weld_scan_2023_08_23_15_23_45/layer_4/'
        # 加载音频文件
        original_path = layer_path + "mic_recording_filter.wav"
        # 使用librosa加载音频文件
        print(layer_path)
        y, sr = librosa.load(original_path, sr=None)

        # 基于采样率计算需要裁剪的样本数
        start_samples = 4 * sr
        end_samples = 2 * sr  # 最后一秒

        # 使用数组切片来裁剪音频
        y_cut = y[start_samples:-end_samples]

        # 获取原始文件的目录，并在此基础上创建输出路径
        output_path = os.path.join(os.path.dirname(original_path), "mic_recording_cut.wav")
        sf.write(output_path, y_cut, sr)

        # 保存处理后的音频
        filepath = layer_path +  "mic_recording_cut.wav"

        # y, sr = sf.read(filepath)
        # 获取文件所在的路径
        audio_directory = os.path.dirname(filepath)

        with wave.open(layer_path + "mic_recording_cut.wav", "rb") as wf:
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
        # plt.show()
        plt.close()
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plot_spectrum(audio_samples, 44000)
        plt.title("Original Signal Spectrum")

        plt.subplot(2, 1, 2)
        plot_spectrum(denoised_samples, 44000)
        plt.title("Denoised Signal Spectrum")

        plt.tight_layout()
        # plt.show()
        plt.close()
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
else:
    print(f"Path '{base_path}' does not exist!")