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
import time

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
base_path = '../data/wall_weld_test/316L_model_130ipm_2023_10_16_22_53_13/'

if os.path.exists(base_path):
    # Get all subdirectories in the specified path
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    # Use regular expression to match layer_n pattern subdirectories
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
        n=0
        while n < 1:
            # Load the audio file
            y, sr = librosa.load(layer_path + 'mic_recording_cut.wav', sr=None)
            # Compute MFCC
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Get 13 MFCC coefficients
            print("MFCCs shape:", mfccs.shape)  # The shape obtained here is usually (13, number of time frames)
            print(mfccs[:,0])
            mfcc_mean = np.mean(mfccs)
            print('mfcc_mean', mfcc_mean)
            
            plt.figure(figsize=(10, 4))
            img = librosa.display.specshow(mfccs, x_axis='time', cmap='viridis')
            plt.ylabel('MFCC Coefficient Index')
            plt.xlabel('Time (frames)')
            plt.title(f'MFCCs of {layer_dir}')
            plt.tight_layout()
            plt.close()

            for i in range(2):
                plt.plot(mfccs[i], label=f'MFCC co {i+1}')
                plt.axhline(y=np.mean(mfccs[i]), color='r', linestyle='-', label=f'MFCC co_mean {i+1}')
                plt.ylabel('MFCC Coefficients')
                plt.xlabel('number of frames')
                plt.title(f'MFCC 1st and 2nd coefficients of {layer_dir}')
            plt.legend()
            plt.show()
            plt.close()
            
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
            ax = plt.gca()  # Get the current axes object
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Force x-axis ticks to be integers
            plt.xlabel(f'Index of samples ({window_length}samples/per)')
            plt.ylabel("MFCC coefficient mean_mov of")
            plt.title(f"Mean_mov of the MFCC of {layer_dir}")
            plt.legend()
            plt.show()
            plt.close()                  
            n += 1  
            # Perform PCA analysis to reduce dimensions (e.g., from 13 dimensions to 2 for visualization)
#             pca = PCA(n_components=2)
#             mfccs_pca = pca.fit_transform(mfccs.T)  # Note to transpose MFCCs because PCA expects samples on rows

#             # Visualize MFCCs after PCA
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
ax = plt.gca()  # Get the current axes object
ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Force x-axis ticks to be integers
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
plt.axhline(y=15, color='blue', linestyle='-', label=f'MFCC co1_mean_thres')
plt.axhline(y=12, color='orange', linestyle='-', label=f'MFCC co2_mean_thres')
ax = plt.gca()  # Get the current axes object
ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Force x-axis ticks to be integers
plt.xlabel('Index of layers')
plt.ylabel("MFCC coefficient mean")
plt.title("Mean of the MFCC")
plt.legend()
plt.show()
plt.close()
