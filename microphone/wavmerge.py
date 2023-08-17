import librosa
import soundfile as sf
import numpy as np

# 1. 加载两个音频文件
y1, sr1 = librosa.load("../data/wall_weld_test/moveL_100_repeat_weld_scan_2023_08_02_17_07_02/layer_1/mic_recording_cut.wav", sr=None)
y2, sr2 = librosa.load("../data/wall_weld_test/moveL_100_repeat_weld_scan_2023_08_02_17_07_02/layer_10/mic_recording_cut.wav", sr=None)

# 确保两个文件有相同的采样率
if sr1 != sr2:
    raise ValueError("The two WAV files have different sample rates!")

# 2. 将它们连接在一起
y_combined = np.concatenate((y1, y2))

# 3. 保存连接后的音频
sf.write("../data/wall_weld_test/moveL_100_repeat_weld_scan_2023_08_02_17_07_02/layer_10/mic_recording_merge.wav", y_combined, sr1)
