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
    
    # If the layer is an odd number, it's in the reverse direction
    # if layer_num % 2 == 1:
    #     previous_profile = previous_profile[::-1]
    
    # Ensure both layers have the same length if they differ
    min_length = min(len(current_profile), len(previous_profile))
    current_profile = current_profile[:min_length]
    previous_profile = previous_profile[:min_length]
    
    dh = current_profile - previous_profile
    return dh

base_path = '../data/wall_weld_test/ER4043_correction_100ipm_2023_09_27_20_53_05/layer_#/'
layer_num = 3
dh = compute_dh(base_path, layer_num)
tip_dis = (np.mean(dh) + 2.3420716473455623) - dh
base_path = '../data/wall_weld_test/ER4043_correction_100ipm_2023_09_27_20_53_05/layer_3/'
fs_wav, audio_data = wav.read(base_path + "mic_recording.wav")
height_path = '../data/wall_weld_test/ER4043_correction_100ipm_2023_09_27_20_53_05/'

df = pd.read_csv(base_path + "current.csv")
time_stamps = df['timestamp'].values
current_signal = df['current'].values

# Load coordinate file
height_profile = np.load(base_path + 'scans/' + "height_profile.npy")
print('print(height_profile.shape):',height_profile.shape)

# Create a time axis with the same length as the current signal
time_axis = np.linspace(0, len(current_signal) / fs_wav, len(current_signal))
print('time_axis:',time_axis.shape)

# Create a new plot
# 1. Calculate the mean value of the current signal
threshold_start = 10
threshold_end = 10

# 2. Find the first point in the current signal that exceeds the threshold
start_index = np.where(current_signal > threshold_start)[0][0]

# 3. Starting from the initial point, find the first point in the current signal below the threshold
end_index = np.where(current_signal[start_index:] < threshold_end)[0][0] + start_index

# 4. Clip the audio signal using these two time points
audio_start_index = int(start_index * len(audio_data) / len(current_signal))
audio_end_index = int(end_index * len(audio_data) / len(current_signal))
audio_clipped = audio_data[audio_start_index:audio_end_index]

# Interpolate the height_profile to have the same length as time_axis
height_interpolated = interp1d(np.linspace(0, 1, len(dh)), dh[:, 1])(np.linspace(0, 1, end_index - start_index))
tip_interpolated = interp1d(np.linspace(0, 1, len(tip_dis)), tip_dis[:, 1])(np.linspace(0, 1, end_index - start_index))

# Extract layer number from base_path
layer_number = int(base_path.split('layer_')[-1].split('/')[0])

# Determine if layer_number is even or odd
if layer_number % 2 == 0:  # Even
    direction = "left-to-right"
else:  # Odd
    direction = "right-to-left"

# Reverse the height_profile if direction is from right to left
if direction == "left-to-right":
    height_interpolated = height_interpolated[::-1]
    tip_interpolated = tip_interpolated[::-1]

# Plot the clipped data
plt.figure(figsize=(10, 8))

# Define the time axis start and end times for the current signal
current_time_start = time_stamps[start_index]
current_time_end = time_stamps[end_index]

# Calculate the time axis for the clipped audio signal
audio_time_stamps = np.linspace(current_time_start, current_time_end, len(audio_clipped))

# Plot clipped audio signal
plt.subplot(3, 1, 1)
plt.plot(audio_time_stamps, audio_clipped)
plt.title('Acoustic Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (Pa)')

# Plot clipped current signal
plt.subplot(3, 1, 2)
plt.plot(time_stamps[start_index:end_index], current_signal[start_index:end_index])
plt.title('Current Signal')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')

# Plot clipped coordinate signal
plt.subplot(3, 1, 3)
plt.plot(time_stamps[start_index:end_index], height_interpolated)
plt.title('Layer by Layer Height Profile')
plt.xlabel('Time (s)')
plt.ylabel('Height (mm)')

plt.tight_layout()
plt.show()

# Saving the clipped current signal as a .npy file in the current directory
clipped_current = current_signal[start_index:end_index]
np.save(base_path + 'current_clipped.npy', clipped_current)
np.save(base_path + 'audio_clipped.npy', audio_clipped)
np.save(base_path + 'height_interpolated.npy', height_interpolated)

# Start and end times
t_start = 5
t_end = t_start + 0.3

# Clip from the audio signal
audio_start_idx = int(t_start * fs_wav)
audio_end_idx = int(t_end * fs_wav)
audio_subset = audio_data[audio_start_idx:audio_end_idx]

# Clip from the current signal
current_subset = df[(df['timestamp'] >= t_start) & (df['timestamp'] <= t_end)]['current'].values
time_subset = df[(df['timestamp'] >= t_start) & (df['timestamp'] <= t_end)]['timestamp'].values

# Plot
plt.figure(figsize=(10, 6))

# Plot clipped audio signal
plt.subplot(2, 1, 1)
plt.plot(np.linspace(t_start, t_end, len(audio_subset)), audio_subset)
plt.title('Audio Signal Subset')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot clipped current signal
plt.subplot(2, 1, 2)
plt.plot(time_subset, current_subset)
plt.title('Current Signal Subset')
plt.xlabel('Time (s)')
plt.ylabel('Current Value')

plt.tight_layout()
plt.show()

# # 1. Calculate energy of audio signal with respect to time
# frame_length = int(fs_wav * 0.02)  # 20 ms frames
# hop_length = int(fs_wav * 0.01)  # 10 ms overlap
# audio_clipped = audio_clipped.astype(np.float32) / np.iinfo(np.int16).max

# # 2. Use the Librosa library to compute the short-time Fourier transform (STFT) of the audio signal
# S = np.abs(librosa.stft(audio_clipped, n_fft=1024, hop_length=hop_length, win_length=frame_length))

# # # 3. Compute the energy of each frame
# energy = np.sum(S ** 2, axis=0)

# # 4. Plot the energy of the audio signal
# plt.figure(figsize=(10, 4))
# plt.plot(np.linspace(current_time_start, current_time_end, len(energy)), energy)
# plt.title('Energy of Audio Signal')
# plt.xlabel('Time (s)')
# plt.ylabel('Energy')
# plt.show()

# 5. Export the clipped audio signal as a WAV file
wav.write(base_path + "mic_recording_clipped.wav", fs_wav, audio_clipped.astype(np.int16))
