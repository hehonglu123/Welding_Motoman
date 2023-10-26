import numpy as np
import wave, copy
import matplotlib.pyplot as plt
import scipy.signal as signal
# import librosa

# Test 1: Signal Plotting Test
def signal_plot_test(audio_signal):
    plt.figure()
    plt.plot(audio_signal)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title('Audio Signal Plot')
    plt.show()

# Test 2: Spectral Analysis Test
def spectral_analysis_test(audio_signal, sample_rate):
    n_samples = len(audio_signal)
    frequencies = np.fft.fftfreq(n_samples, d=1.0/sample_rate)
    magnitude_spectrum = np.abs(np.fft.fft(audio_signal))

    plt.figure()
    plt.plot(frequencies[:n_samples//2], magnitude_spectrum[:n_samples//2])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude Spectrum')
    plt.title('Frequency Spectrum using FFT')
    plt.show()

# Test 3: Spectrogram with Actual Time on X-axis
def spectrogram_test(audio_signal, sample_rate):
    f, t, Sxx = signal.spectrogram(audio_signal, sample_rate)
    mean_freq = np.sum(Sxx * np.arange(Sxx.shape[0])[:, np.newaxis], axis=0) / np.sum(Sxx, axis=0)
    plt.figure()
    plt.pcolormesh(t, f,np.log(Sxx))
    plt.colorbar(label='Log Magnitude')
    plt.ylim(0, 5000)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram Log')
    plt.show()

# Test 4: Amplitude Envelope Test
def amplitude_envelope_test(audio_signal):
    amplitude_envelope = np.abs(signal.hilbert(audio_signal))
    plt.figure()
    plt.plot(audio_signal, label='Audio Signal')
    plt.plot(amplitude_envelope, label='Amplitude Envelope')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title('Amplitude Envelope')
    plt.legend()
    plt.show()

wavfile = wave.open('../data/wall_weld_test/ER4043_correction_100ipm_2023_09_27_20_53_05/layer_7/mic_recording_cut.wav', 'rb')

samplerate = 44000
channels = 1
audio_data=np.frombuffer(wavfile.readframes(wavfile.getnframes()),dtype=np.int16)
signal_plot_test(audio_data)
spectral_analysis_test(audio_data, samplerate)
spectrogram_test(audio_data, samplerate)
amplitude_envelope_test(audio_data)
