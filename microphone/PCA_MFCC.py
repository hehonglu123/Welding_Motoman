import librosa
import librosa.display
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.decomposition import PCA
import sys

n=10
while n < 16:
# 加载音频文件
    y, sr = librosa.load(f'../data/wall_weld_test/moveL_100_repeat_weld_scan_2023_08_02_17_07_02/layer_{n}/mic_recording_cut.wav', sr=None)

    # 计算MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 获取13个MFCC系数
    print("MFCCs shape:", mfccs.shape)  # 此处得到的形状通常为(13, 时间帧数)
    # plt.figure(figsize=(10, 4))
    # img = librosa.display.specshow(mfccs, x_axis='time', cmap='viridis')
    # plt.colorbar(img, label='MFCC Coefficient Value')
    # plt.ylabel('MFCC Coefficient Index')
    # plt.xlabel('Time (frames)')
    # plt.title('MFCCs')
    # plt.tight_layout()
    # plt.show()
    # 进行PCA分析，以减少维度（例如，从13维减少到2维以便于可视化）
    pca = PCA(n_components=2)
    mfccs_pca = pca.fit_transform(mfccs.T)  # 注意要转置MFCCs，因为PCA期望样本在行上

    # 可视化PCA处理后的MFCC
    plt.scatter(mfccs_pca[:, 0], mfccs_pca[:, 1], edgecolor='red', alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.xlim([-200,400])
    plt.ylabel('Principal Component 2')
    plt.ylim([-60,150])
    plt.title(f'PCA of MFCCs in segments {n+1}')
    plt.show()
    n += 1