import numpy as np
import scipy.fftpack
import librosa
import librosa.display
import matplotlib.pyplot as plt

file_path = '../data/wall_weld_test/ER4043_correction_100ipm_2023_09_27_20_53_05/layer_5/'

y, sr = librosa.load(file_path + "mic_recording.wav", sr=None)  # sr=None to preserve original sampling rate

# Mean
mean_amplitude = np.mean(np.abs(y))

# Variance
variance_amplitude = np.var(y)

# Root Mean Square Energy
rmse = np.sqrt(np.mean(y**2))

print(f"Mean Amplitude: {mean_amplitude}")
print(f"Variance: {variance_amplitude}")
print(f"RMSE: {rmse}")


Y = scipy.fftpack.fft(y)
frequencies = np.linspace(0, sr, len(Y))

# Plot the spectrum
plt.figure(figsize=(10, 4))
plt.semilogy(frequencies, np.abs(Y))
plt.title('Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()

# Main frequency component
main_frequency = frequencies[np.argmax(np.abs(Y))]

print(f"Main Frequency: {main_frequency} Hz")


plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.figure(figsize=(10, 4))
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()

# Compute short-time energy using a sliding window
frame_length = 1024
hop_length = 512
energy = np.array([
    sum(abs(y[i:i+frame_length]**2))
    for i in range(0, len(y), hop_length)
])

# Plot
plt.figure(figsize=(10, 4))
times = np.arange(len(energy)) * hop_length / sr
plt.plot(times, energy)
plt.ylabel('Energy')
plt.xlabel('Time (s)')
plt.title('Short-time Energy')
plt.tight_layout()
plt.show()

rmse = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)

plt.figure(figsize=(10, 4))
times = librosa.frames_to_time(np.arange(len(rmse[0])), sr=sr, hop_length=hop_length)
plt.plot(times, rmse[0])
plt.ylabel('RMSE')
plt.xlabel('Time (s)')
plt.title('Root Mean Square Energy (RMSE)')
plt.tight_layout()
plt.show()