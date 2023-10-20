import librosa
import librosa.display
import soundfile as sf
from scipy.signal import butter, lfilter
import os, re, pywt, wave, copy, warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning, module="lazy_loader")
def highpass_filter(data, sr, cutoff=1000):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    y_highpassed = lfilter(b, a, data)
    return y_highpassed

def wavelet_denoise(data, wavelet='db1', level=4, threshold_type='soft'):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    coeffs_thresholded = [pywt.threshold(c, threshold, mode=threshold_type) for c in coeffs]
    return pywt.waverec(coeffs_thresholded, wavelet)

def plot_spectrum(samples, fs):
    N = len(samples)
    yf = np.fft.fft(samples)
    xf = np.fft.fftfreq(N, 1/fs)
    plt.plot(xf[:N//2], 2.0/N * np.abs(yf[:N//2]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()

def audio_denoise(file_path):
    if os.path.isfile(file_path + "mic_recording.wav"):
        print('Searching data path in: ',file_path)
        file_path = copy.deepcopy(file_path) + "mic_recording.wav"
        y, sr = librosa.load(file_path, sr=None)
        
        y_highpassed = highpass_filter(y, sr)
        sf.write(file_path.replace(".wav", "_filter.wav"), y_highpassed, sr)
        
        y_cut = y_highpassed[int(4*sr):-int(2*sr)]
        sf.write(file_path.replace(".wav", "_cut.wav"), y_cut, sr)

        with wave.open(file_path.replace(".wav", "_cut.wav"), "rb") as wf:
            n_samples = wf.getnframes()
            audio_data = wf.readframes(n_samples)
            audio_samples = np.frombuffer(audio_data, dtype=np.int16)
            denoised_samples = wavelet_denoise(audio_samples)

            # Saving denoised file
            sf.write(file_path.replace(".wav", "_denoised.wav"), denoised_samples, 44000)

            output_folder = os.path.join(os.path.dirname(file_path), "microphone_segments")
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            segment_length = len(denoised_samples) // 20
            for i in range(20):
                segment = denoised_samples[i*segment_length: (i+1)*segment_length]
                sf.write(os.path.join(output_folder, f'segments_{i}.wav'), segment, 44000)

    else:
        print(f"File '{file_path}' does not exist!")

# Example usage
# audio_denoise("path_to_file.wav")

def audio_MFCC(file_path):

    if os.path.isfile(file_path + "mic_recording_cut.wav"):
        print('Searching data path in: ',file_path)

        mic_recording_path = os.path.join(file_path, "mic_recording_cut.wav")

        if not os.path.exists(mic_recording_path):
            print(f"mic_recording_cut.wav not found in {file_path}. Skipping...")
            
        y, sr = librosa.load(file_path + 'mic_recording_cut.wav', sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # print("MFCCs shape:", mfccs.shape)
        # print(mfccs[:, 0])
        

        std_value_co1 = np.std(mfccs[0])
        print('std_value_co1:',std_value_co1)
        std_value_co2 = np.std(mfccs[1])
        print('std_value_co2:',std_value_co2)
        if std_value_co1 > 15 or std_value_co2 > 12:
            scan_flag = True
        else:
            scan_flag = False
        
        return std_value_co1, std_value_co2, scan_flag

    else:
        print(f"Path '{file_path}' does not exist!")
        
if __name__ == "__main__":
    data_dir='../data/wall_weld_test/316L_model_130ipm_2023_10_16_22_53_13/'
    if os.path.exists(data_dir):
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        layer_dirs = [d for d in subdirs if re.match(r'layer_\d+', d)]
        for layer_dir in sorted(layer_dirs, key=lambda x: int(x.split('_')[-1])):
            layer_path = os.path.join(data_dir, layer_dir + '/',)
            mic_recording_path = os.path.join(layer_path, "mic_recording.wav")
            if not os.path.exists(mic_recording_path):
                print(f"mic_recording.wav not found in {layer_path}. Skipping...")
                continue  # Skip to the next iteration
            audio_denoise(layer_path)
            std_value_co1, std_value_co2, scan_flag = audio_MFCC(layer_path)    