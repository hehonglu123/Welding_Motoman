import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 加载音频文件
y, sr = librosa.load('path_to_audio_file.wav', sr=None)

# 计算MFCC
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 获取13个MFCC系数
print("MFCCs shape:", mfccs.shape)  # 此处得到的形状通常为(13, 时间帧数)

# 进行PCA分析，以减少维度（例如，从13维减少到2维以便于可视化）
pca = PCA(n_components=2)
mfccs_pca = pca.fit_transform(mfccs.T)  # 注意要转置MFCCs，因为PCA期望样本在行上

# 可视化PCA处理后的MFCC
plt.scatter(mfccs_pca[:, 0], mfccs_pca[:, 1], edgecolor='none', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of MFCCs')
plt.show()