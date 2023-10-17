import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.decomposition import PCA
import sys
import os
import re
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation
from moviepy.editor import VideoClip

def moving_average(data_list, window_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(data_list, weights, mode='valid')
mean_mov_co1 = []
mean_mov_co2 = []
mfcc_mean = []
std_co1 = []
std_co2 = []
std_value_co1 = []
std_value_co2 = []
mean_co1 = []
mean_co2 = []
mean_value_co1 = []
mean_value_co2 = []
window_length = []
base_path = '../data/wall_weld_test/ER4043_correction_100ipm_2023_09_27_20_53_05/'

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
            std_co1.append(0)
            std_co2.append(0)
            mean_co1.append(0)
            mean_co2.append(0) 
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
            mfcc_mean = np.mean(mfccs)
            print('mfcc_mean', mfcc_mean)
            
            plt.figure(figsize=(10, 4))
            img = librosa.display.specshow(mfccs, x_axis='time', cmap='viridis')
            # plt.colorbar(img, label='MFCC Coefficient Value')
            plt.ylabel('MFCC Coefficient Index')
            plt.xlabel('Time (frames)')
            plt.title(f'MFCCs of {layer_dir}')
            plt.tight_layout()
            # plt.show()
            plt.close()
            fig, ax = plt.subplots()

            # 设置y轴的范围为两组mfccs数据的最小值和最大值
            ax.set_ylim(min(np.min(mfccs[0]), np.min(mfccs[1])) - 1, max(np.max(mfccs[0]), np.max(mfccs[1])) + 1)

            # 创建两个空的数据集来收集x和y的数据
            xdata1, ydata1, xdata2, ydata2 = [], [], [], []
            ln2, = ax.plot([], [], 'b-', animated=True, label='MFCC Coefficient 2')
            ln1, = ax.plot([], [], 'g-', animated=True, label='MFCC Coefficient 1')

            def init():
                ax.set_xlim(0, max(len(mfccs[0]), len(mfccs[1])))
                ax.set_ylabel('MFCC Coefficients')
                ax.set_xlabel('number of frames')
                ax.set_title(f'MFCC 1st and 2nd coefficients of {layer_dir}')
                ax.legend(loc = 'lower right')
                return ln1, ln2

            def update(frame):
                if frame < len(mfccs[0]):
                    xdata1.append(frame)
                    ydata1.append(mfccs[0][frame])
                    ln1.set_data(xdata1, ydata1)
                if frame < len(mfccs[1]):
                    xdata2.append(frame)
                    ydata2.append(mfccs[1][frame])
                    ln2.set_data(xdata2, ydata2)
                return ln1, ln2

            total_time_milliseconds = 5000
            total_frames = max(len(mfccs[0]), len(mfccs[1]))

            interval_time = total_time_milliseconds / total_frames
            
            ani = FuncAnimation(fig, update, frames=range(total_frames), 
                                init_func=init, blit=True, repeat=False, 
                                interval=interval_time)

            plt.show()
            # 将matplotlib动画转换为moviepy的VideoClip对象
            duration = total_frames * (5000 / total_frames) / 1000.0
            video_clip = VideoClip(lambda x: ani.to_rgba(x, bytes=True, norm=True), duration=duration)

            # 保存为MP4
            video_clip.write_videofile(f'{layer_dir}.mp4', fps=30)
            plt.close()
            # 如果你想要显示图例，可以使用以下命令：
            # plt.legend()
            # plt.show()
            # plt.close()
            std_value_co1 = np.std(mfccs[0])
            print('std_value_co1:',std_value_co1)
            std_value_co2 = np.std(mfccs[1])
            print('std_value_co2:',std_value_co2)
            std_co1.append(std_value_co1)
            std_co2.append(std_value_co2)
            mean_value_co1 = np.mean(mfccs[0])
            mean_value_co2 = np.mean(mfccs[1])
            mean_co1.append(mean_value_co1)
            mean_co2.append(mean_value_co2)  
            window_length = int(mfccs.shape[1]/40)
            mean_mov_co1 = moving_average(mfccs[0],window_length)
            mean_mov_co2 = moving_average(mfccs[1],window_length)   
            x_labels = range(len(mean_mov_co1))
            plt.figure(figsize=(6, 6))
            plt.plot(x_labels, mean_mov_co1, marker='o', linestyle='-',color="blue", label = 'mean_mov_co1')
            plt.plot(x_labels, mean_mov_co2, marker='o', linestyle='-',color="orange",label = 'mean_mov_co2')
            ax = plt.gca()  # 获取当前的axes对象
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # 强制x轴刻度为整数
            plt.xlabel(f'Index of samples ({window_length}samples/per)')
            plt.ylabel("MFCC coefficient mean_mov of")
            plt.title(f"Mean_mov of the MFCC of {layer_dir}")
            plt.legend()
            # plt.show()
            plt.close()                  
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
plt.axhline(y=15, color='blue', linestyle='-', label=f'MFCC co1_std_thres')
plt.axhline(y=12, color='orange', linestyle='-', label=f'MFCC co2_std_thres')
ax = plt.gca()  # 获取当前的axes对象
ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # 强制x轴刻度为整数
plt.xlabel('Index of layers')
plt.ylabel("MFCC coefficient standard deviation")
plt.title("Standard Deviation of the MFCC")
plt.legend()
plt.show()
plt.close()

x_labels = range(len(mean_co1))
plt.figure(figsize=(6, 6))
plt.plot(x_labels, mean_co1, marker='o', linestyle='-',color="blue", label = 'mean_co1')
plt.plot(x_labels, mean_co2, marker='o', linestyle='-',color="orange",label = 'mean_co2')
ax = plt.gca()  # 获取当前的axes对象
ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # 强制x轴刻度为整数
plt.xlabel('Index of layers')
plt.ylabel("MFCC coefficient mean")
plt.title("Mean of the MFCC")
plt.legend()
plt.show()
plt.close()




