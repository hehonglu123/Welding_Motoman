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
    # Decompose the signal using wavelets
    coeffs = pywt.wavedec(data, wavelet, level=level)
    # Estimate noise
    sigma = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    # Threshold coefficients
    coeffs_thresholded = [pywt.threshold(c, threshold, mode=threshold_type) for c in coeffs]
    # Reconstruct the denoised signal
    return pywt.waverec(coeffs_thresholded, wavelet)

# Compute the spectrum of the audio
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
    # Get all subdirectories in the specified path
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    # Use regex to match subdirectories with pattern 'layer_n'
    layer_dirs = [d for d in subdirs if re.match(r'layer_\d+', d)]

    for layer_dir in sorted(layer_dirs, key=lambda x: int(x.split('_')[-1])):
        layer_path = os.path.join(base_path, layer_dir + '/',)
        # Construct the path to the mic_recording.wav file
        mic_recording_path = os.path.join(layer_path, "mic_recording_filter.wav")
        
        # Check if mic_recording.wav exists in the current subdir
        if not os.path.exists(mic_recording_path):
            print(f"mic_recording_filter.wav not found in {layer_path}. Skipping...")
            continue  # Skip to the next iteration
        
        print(layer_path)
        y, sr = librosa.load(mic_recording_path, sr=None)

        # Compute the number of samples to cut based on the sampling rate
        start_samples = 4 * sr
        end_samples = 2 * sr  # last second

        # Use array slicing to cut the audio
        y_cut = y[start_samples:-end_samples]

        # Create an output path based on the directory of the original file
        output_path = os.path.join(os.path.dirname(mic_recording_path), "mic_recording_cut.wav")
        sf.write(output_path, y_cut, sr)

        # Read the processed audio
        with wave.open(output_path, "rb") as wf:
            n_samples = wf.getnframes()
            audio_data = wf.readframes(n_samples)
            audio_samples = np.frombuffer(audio_data, dtype=np.int16)

        fs = 44000  # sampling rate
        t = np.arange(len(audio_samples)) / fs  # time axis

        # Denoise the audio samples
        denoised_samples = wavelet_denoise(audio_samples)

        # Recompute the time axis for the denoised samples
        t_denoised = np.arange(len(denoised_samples)) / 44000

        # Plot original and denoised signal
        t_original = np.arange(len(audio_samples)) / 44000  # time axis for original signal

        # [... Rest of the plotting code ...]

        # Create a new folder in the path of the audio file
        output_folder = os.path.join(layer_path, "microphone_segments")
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        # Calculate the length of each segment
        segment_length = len(audio_samples) // 20

        # Segment the audio and save
        for i in range(20):
            segment = audio_samples[i*segment_length: (i+1)*segment_length]
            sf.write(os.path.join(output_folder, f'segments_{i}.wav'), segment, fs)
else:
    print(f"Path '{base_path}' does not exist!")
