import librosa
import soundfile as sf
import numpy as np

# 1. Load two audio files
y1, sr1 = librosa.load("../data/wall_weld_test/moveL_100_repeat_weld_scan_2023_08_02_17_07_02/layer_1/mic_recording_cut.wav", sr=None)
y2, sr2 = librosa.load("../data/wall_weld_test/moveL_100_repeat_weld_scan_2023_08_02_17_07_02/layer_10/mic_recording_cut.wav", sr=None)

# Ensure the two files have the same sample rate
if sr1 != sr2:
    raise ValueError("The two WAV files have different sample rates!")

# 2. Concatenate them together
y_combined = np.concatenate((y1, y2))

# 3. Save the combined audio
sf.write("../data/wall_weld_test/moveL_100_repeat_weld_scan_2023_08_02_17_07_02/layer_10/mic_recording_merge.wav", y_combined, sr1)
