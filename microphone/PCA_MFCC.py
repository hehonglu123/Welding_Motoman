import librosa
import librosa.display
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.decomposition import PCA
import sys
import os
import re
from matplotlib.ticker import MaxNLocator

std_co1 = []
std_co2 = []
std_value_co1 = []
std_value_co2 = []
base_path = '../data/wall_weld_test/weld_scan_correction_2023_09_19_21_14_58/'

if os.path.exists(base_path):
    # 获取指定路径下的所有子目录
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    # 使用正则表达式匹配 layer_n 模式的子目录
    layer_dirs = [d for d in subdirs if re.match(r'layer_\d+', d)]

    for layer_dir in sorted(layer_dirs, key=lambda x: int(x.split('_')[-1])):
        layer_path = os.path.join(base_path, layer_dir + '/',)

        # Construct the path to the mic_recording.wav file
        mic_recording_path = os.path.join(layer_path, "mic_recording_cut.wav")
        
        # Check if mic_recording.wav exists in the current subdir
        if not os.path.exists(mic_recording_path):
            print(f"mic_recording_cut.wav not found in {layer_path}. Skipping...")
            continue  # Skip to the next iteration
# file_path = '../data/wall_weld_test/moveL_100_repeat_weld_scan_2023_08_02_17_07_02/layer_0/'
        n=0
        while n < 1:
        # 加载音频文件
            y, sr = librosa.load(layer_path + 'mic_recording_cut.wav', sr=None)

            # 计算MFCC
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 获取13个MFCC系数
            print("MFCCs shape:", mfccs.shape)  # 此处得到的形状通常为(13, 时间帧数)
            print(mfccs[:,0])
            plt.figure(figsize=(10, 4))
            img = librosa.display.specshow(mfccs, x_axis='time', cmap='viridis')
            # plt.colorbar(img, label='MFCC Coefficient Value')
            plt.ylabel('MFCC Coefficient Index')
            plt.xlabel('Time (frames)')
            plt.title(f'MFCCs of {layer_dir}')
            plt.tight_layout()
            # plt.show()
            plt.close()

            for i in range(2):
                plt.plot(mfccs[i], label=f'MFCC co {i+1}')
                plt.ylabel('MFCC Coefficients')
                plt.xlabel('number of frames')
                plt.title(f'MFCC 1st and 2nd coefficients of {layer_dir}')

            # 如果你想要显示图例，可以使用以下命令：
            # plt.legend()

            plt.show()
            plt.close()
            std_value_co1 = np.std(mfccs[0])
            std_value_co2 = np.std(mfccs[1])
            std_co1.append(std_value_co1)
            std_co2.append(std_value_co2)
            n += 1  
            # exit()
#             # 进行PCA分析，以减少维度（例如，从13维减少到2维以便于可视化）
#             pca = PCA(n_components=2)
#             mfccs_pca = pca.fit_transform(mfccs.T)  # 注意要转置MFCCs，因为PCA期望样本在行上

#             # 可视化PCA处理后的MFCC
#             plt.scatter(mfccs_pca[:, 0], mfccs_pca[:, 1], edgecolor='red', alpha=0.7)
#             plt.xlabel('Principal Component 1')
#             plt.xlim([-200,400])
#             plt.ylabel('Principal Component 2')
#             plt.ylim([-200,200])
#             plt.title(f'PCA of MFCCs in segments {n+1}')
#             plt.show()
else:
    print(f"Path '{base_path}' does not exist!")
x_labels = range(len(std_co1))
plt.figure(figsize=(6, 6))
plt.plot(x_labels, std_co1, marker='o', linestyle='-',color="blue", label = 'std_co1')
plt.plot(x_labels, std_co2, marker='o', linestyle='-',color="orange",label = 'std_co2')
ax = plt.gca()  # 获取当前的axes对象
ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # 强制x轴刻度为整数
plt.xlabel('Index of layers')
plt.ylabel("MFCC coefficient standard deviation")
plt.title("Standard Deviation of the MFCC")
plt.legend()
plt.show()
plt.close()