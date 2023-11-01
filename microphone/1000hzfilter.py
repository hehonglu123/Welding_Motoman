import numpy as np
import scipy.signal
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import butter, lfilter
import os
import re

base_path = '../data/wall_weld_test/moveL_100_baseline_weld_scan_2023_07_07_15_20_56/'

if os.path.exists(base_path):
    # Get all subdirectories under the specified path
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    # Use regular expression to match subdirectories with pattern "layer_n"
    layer_dirs = [d for d in subdirs if re.match(r'layer_\d+', d)]

    for layer_dir in sorted(layer_dirs, key=lambda x: int(x.split('_')[-1])):
        layer_path = os.path.join(base_path, layer_dir + '/',)
        # Construct the path to the mic_recording.wav file
        mic_recording_path = os.path.join(layer_path, "mic_recording.wav")
        
        # Check if mic_recording.wav exists in the current subdir
        if not os.path.exists(mic_recording_path):
            print(f"mic_recording.wav not found in {layer_path}. Skipping...")
            continue  # Skip to the next iteration
        
        # Load audio file
        y, sr = librosa.load(layer_path + "mic_recording.wav", sr=None)

        # Design a low-pass filter (commented-out section)
        # nyquist = 0.5 * sr
        # cutoff = 1000  # Desired cutoff frequency, in Hz
        # normal_cutoff = cutoff / nyquist
        # b, a = scipy.signal.butter(6, normal_cutoff, btype='low', analog=False)
        # Apply the filter (commented-out section)
        # y_filtered = scipy.signal.filtfilt(b, a, y)
        # FFT validation (commented-out section)
        # D_original = np.abs(librosa.stft(y))
        # D_filtered = np.abs(librosa.stft(y_filtered))

        # Design a high-pass filter
        def highpass_filter(data, sr, cutoff=1000):
            nyq = 0.5 * sr
            normal_cutoff = cutoff / nyq
            b, a = butter(1, normal_cutoff, btype='high', analog=False)
            y_highpassed = lfilter(b, a, data)
            return y_highpassed

        # Filter out signals below 1000Hz
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
        # plt.show()
        plt.close()

        # Calculate STFT
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
        # plt.show()
        # Save the processed audio
        sf.write(layer_path + "mic_recording_filter.wav", y_highpassed, sr)
        plt.close()

        # Original and Filtered Spectrum (commented-out section)
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

        # Save the filtered audio to a new file (commented-out section)
        # output_path = "../data/wall_weld_test/moveL_100_repeat_weld_scan_2023_08_02_17_07_02/layer_0/mic_recording_filter.wav"
        # sf.write(output_path, y_filtered, sr)

else:
    print(f"Path '{base_path}' does not exist!")
