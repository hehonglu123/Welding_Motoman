import cv2,wave,copy
import pickle, sys
import numpy as np
import scipy.signal as signal
sys.path.append('../toolbox/')
from flir_toolbox import *
import matplotlib.pyplot as plt

for n in range(0,23):
 
    data_dir=f'../data/wall_weld_test/ER4043_model_150ipm_2023_10_08_08_23_41/layer_{n}/'

    fig, ax1 = plt.subplots()
    ax1.set_title(f'{data_dir}')
    ax2 = ax1.twinx()
    ax3 = ax2.twiny()
    # Load the IR recording data from the pickle file
    with open(data_dir+'ir_recording.pickle', 'rb') as file:
        ir_recording = pickle.load(file)
    ir_ts=np.loadtxt(data_dir+'ir_stamps.csv', delimiter=',')
    ir_ts=ir_ts-ir_ts[0]
    center_brightness=np.average(ir_recording[:,100:140,140:180],axis=(1,2))
    # ax1.plot(ir_ts, center_brightness, c='red', label='ir center counts')
    ax1.set_ylabel('Center Brightness (counts)')
    ax1.set_xlabel('length (mm)')

    ##microphone data
    wavfile = wave.open(f'../data/wall_weld_test/ER4043_model_150ipm_2023_10_08_08_23_41/layer_{n}/mic_recording.wav', 'rb')
    samplerate = 44000
    channels = 1
    audio_data=np.frombuffer(wavfile.readframes(wavfile.getnframes()),dtype=np.int16)
    freq, mic_ts, Sxx = signal.spectrogram(audio_data, samplerate)
    mean_freq = np.sum(Sxx * np.arange(Sxx.shape[0])[:, np.newaxis], axis=0) / np.sum(Sxx, axis=0)
    # 确定时间范围索引
    start_index = np.searchsorted(mic_ts, mic_ts[0] + 4)  # 找到4秒后的索引
    end_index = np.searchsorted(mic_ts, mic_ts[-1] - 1)   # 找到结束前2秒的索引

    # 筛选时间戳和平均频率
    selected_mic_ts = mic_ts[start_index:end_index]
    selected_mean_freq = mean_freq[start_index:end_index]

    # 筛选出mean_freq大于15的部分
    selected_freq_indices = np.flatnonzero(selected_mean_freq > 10)

    # 根据筛选结果确定焊接起始和结束时间
    if len(selected_freq_indices) > 0:
        start_welding_mic = selected_mic_ts[selected_freq_indices[0]]
        end_welding_mic = selected_mic_ts[selected_freq_indices[-1]]
        start_time_sec = start_welding_mic  
        end_time_sec = end_welding_mic  
        start_sample_index = int(start_time_sec * samplerate)
        end_sample_index = int(end_time_sec * samplerate)
        welding_duration_mic = end_welding_mic - start_welding_mic
        welding_duration_mic_ext=mic_ts[-1]-mic_ts[0]
        welding_audio_data = audio_data[start_sample_index:end_sample_index]
        # print(welding_audio_data)
    else:
        # 如果没有找到合适的频率，可能需要其他逻辑来处理
        print("No signal captured")
    print('------------------------------------------------')
    print(f'welding signal of {n} layer:')
    print('welding started:', start_welding_mic)
    print('welding ended:', end_welding_mic)
    print('welding duration time:', welding_duration_mic)
    freq_welding, mic_ts_welding, Sxx_welding = signal.spectrogram(welding_audio_data, samplerate)
    selected_mic_ts_welding = mic_ts_welding[start_index:end_index]

    mean_freq_welding = np.sum(Sxx_welding * np.arange(Sxx_welding.shape[0])[:, np.newaxis], axis=0) / np.sum(Sxx_welding, axis=0)
    ax2.plot(mic_ts_welding, mean_freq_welding, label='microphone frequency')
    ax2.set_xlabel('Time Frame')
    ax2.set_ylabel('Frequency, Height Difference')
    # plt.show()

    # ##welding data
    # welding_data=np.loadtxt(data_dir+'welding.csv',skiprows=1, delimiter=',')
    # welding_data[:,0]-=welding_data[0,0]
    # welding_data[:,0]/=1e6
    # ax2.plot(welding_data[:,0],welding_data[:,1],label='voltage')
    # ax2.plot(welding_data[:,0],welding_data[:,2],label='current')
    # ax2.plot(welding_data[:,0],welding_data[:,3],c='gray',label='feedrate')
    # plt.show() 

    ##height profile
    profile_height_current = np.load(data_dir + 'scans/height_profile.npy')
    # Load the previous layer's height profile if n > 1
    if n > 0:
        profile_height_previous = np.load(data_dir.replace(f'layer_{n}/', f'layer_{n-1}/') + 'scans/height_profile.npy')
        # Ensure both arrays are of the same length
        min_length = min(len(profile_height_current), len(profile_height_previous))
        profile_height_current = profile_height_current[:min_length]
        profile_height_previous = profile_height_previous[:min_length]
        # Calculate the difference in height
        height_difference = profile_height_current[:, 1] - profile_height_previous[:, 1]
    else:
        # For the first layer, there is no previous layer to compare to
        height_difference = profile_height_current[:, 1]  # Default to current height
   
    welding_length = profile_height_current[-1, 0] - profile_height_current[0, 0]
    welding_length_ext=welding_length*welding_duration_mic_ext/welding_duration_mic
    welding_duration_mic_ext = welding_length * welding_duration_mic_ext / welding_duration_mic
    welding_ext_start = profile_height_current[0, 0] - welding_length_ext * start_welding_mic / welding_duration_mic_ext

    # ax3.set_xlim(welding_ext_start, welding_ext_start + welding_length_ext)
    # ax1.set_xlim(mic_ts[0], mic_ts[-1])
    # # Plot the height difference instead of the absolute height
    # ax3.scatter(profile_height_current[:, 0], height_difference, c='silver', label='height difference')
    # ax3.set_xlabel('Distance (mm)')

    # h1, l1 = ax1.get_legend_handles_labels()
    # h2, l2 = ax2.get_legend_handles_labels()
    # h3, l3 = ax3.get_legend_handles_labels()
    # ax1.legend(h1 + h2 + h3, l1 + l2 + l3, loc=1)
    # plt.show()

    # 绘制层高差异数据
    ax3.set_xlim(profile_height_current[0, 0], profile_height_current[-1, 0])
    ax1.set_xlim(mic_ts_welding[0], mic_ts_welding[-1])
    ax3.scatter(profile_height_current[:, 0], height_difference, c='silver', label='Height Difference During Welding')
    ax3.set_xlabel('Length (mm)')
    
    # ax1.xlim(welding_ext_start, welding_ext_start + welding_length_ext)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax1.legend(h1 + h2 + h3, l1 + l2 + l3, loc=1)
    ax1.legend()
    plt.show()

    # profile_height = np.load(data_dir + 'scans/height_profile.npy')

    # # 计算每段的长度
    # num_segments = 20
    # segment_length = len(profile_height) // num_segments

    # color_cycle = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999"]
    # colors = [color_cycle[i % len(color_cycle)] for i in range(num_segments)]

    # plt.figure(figsize=(10, 6))

    # # 循环遍历每段并绘制
    # for i in range(num_segments):
    #     start_index = i * segment_length
    #     end_index = (i + 1) * segment_length if i != num_segments - 1 else None  # 如果是最后一段，取到末尾
        
    #     segment_data = profile_height[start_index:end_index]
    #     plt.plot(segment_data[:, 0], segment_data[:, 1], color=colors[i], label='Height Profile' if i == 0 else "")

    # # 如果您只想为一个段加上legend，您可以在上述循环中只为一个段添加label。如上例中，只为第一个segment设置了label。

    # plt.xlabel('Distance (mm)')
    # plt.ylabel('Height')
    # plt.title('Height Profile with Segments')
    # plt.legend()
    # # plt.show()