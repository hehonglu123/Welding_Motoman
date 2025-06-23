import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import sys, datetime, yaml, pathlib, glob, os
sys.path.append('../../mocap/')
from Models import *

np.random.seed(42) # for reproducibility
torch.manual_seed(42) # for reproducibility

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(data_dir, input_size, hidden_size, output_size, num_layers, epochs, batch_size, learning_rate):
    
    # load data
    train_data_dir = data_dir[:-1]
    test_data_dir = data_dir[:-1]
    train_data = []
    for dir_name in train_data_dir:
        train_data.append(np.loadtxt(dir_name+'profile_welding_10_dhdw.csv', delimiter=',', skiprows=1))
    test_data = []
    for dir_name in test_data_dir:
        test_data.append(np.loadtxt(dir_name+'profile_welding_10_dhdw.csv', delimiter=',', skiprows=1))
    

    # model
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers).to(device)
    # loss function
    loss_fn = nn.MSELoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == "__main__":

    # load data
    geo_data_dir = '../../data/wall_weld_test/'
    logdata_dir_all = ['weld_fujiscan_2025_06_11_16_27_41/','weld_fujiscan_2025_06_11_16_52_36/','weld_fujiscan_2025_06_11_17_16_48/',\
                       'weld_fujiscan_2025_06_11_17_49_27/','weld_fujiscan_2025_06_11_18_14_56/','weld_fujiscan_2025_06_12_17_33_24/',\
                       'weld_fujiscan_2025_06_12_16_59_09/','weld_fujiscan_2025_06_12_15_33_03/','weld_fujiscan_2025_06_12_15_03_27/']

    # parameters
    sample_rate = 10 # Hz, using the rate of ir camera
    train_test_split = 0.8 # 80% for training, 20% for testing
    epochs = 10000 # number of epochs for training
    batch_size = 50 # batch size for training
    learning_rate = 0.001 # learning rate for training

    # model parameters
    model_input_size = 4 # cmd_v, cmd_fd, dh, dw
    model_hidden_size = 64 # hidden size of the LSTM
    lstm_num_layers = 1 # number of layers in the LSTM
    model_output_size = 2 # dh, dw

    ignore_start_end = 5
    start_x = -55 + ignore_start_end
    end_x = 55 - ignore_start_end

    ### data processing
    data_dirs = []
    train_data = []
    train_data_batch_len = []
    for i, logdata_dir in enumerate(logdata_dir_all):
        total_layers_name = glob.glob(geo_data_dir+logdata_dir+'layer*')
        # get printed layer number
        layer_nums = []
        for layer_name in total_layers_name:
            this_layer = layer_name.split('\\')[-1]
            this_layer = this_layer.split('r')[-1]
            layer_nums.append(int(this_layer))
        layer_nums = np.sort(layer_nums)
        for layer_n_id, layer_n in enumerate(layer_nums):
            layer_name = 'layer'+str(layer_n)
            this_layer_dir = geo_data_dir + logdata_dir + layer_name + '/'
            if os.path.exists(this_layer_dir+'profile_welding_'+str(sample_rate)+'_dhdw.csv'):
                data_dirs.append(this_layer_dir)
                train_data.append(np.loadtxt(this_layer_dir+'profile_welding_'+str(sample_rate)+'_dhdw.csv', delimiter=',', skiprows=1))
                train_data_batch_len.append(len(train_data[-1]))
            else:
                print("Processing layer "+str(layer_n)+" in "+logdata_dir)
                print('No data for layer '+str(layer_n)+' in '+logdata_dir)
                print("Interpolate data from profile welding.csv")
                # load data
                profile_welding = np.loadtxt(this_layer_dir+'profile_welding.csv', delimiter=',', skiprows=1)
                # chop x < start_x or x > end_x
                x_location = np.array(profile_welding[:, 1])
                if x_location[-1]>x_location[0]:
                    start_index = np.where(x_location > start_x)[0][0] 
                    end_index = np.where(x_location < end_x)[0][-1]
                else:
                    start_index = np.where(x_location < end_x)[0][0] 
                    end_index = np.where(x_location > start_x)[0][-1]
                profile_welding = profile_welding[start_index:end_index+1, :]
                # interpolate the data to the sample rate
                timestamp_welding = profile_welding[:,0]
                timestamps_interp = np.arange(timestamp_welding[0]+1/sample_rate, timestamp_welding[-1], 1/sample_rate)
                cmd_v_interp = np.zeros_like(timestamps_interp)
                cmd_fd_interp = np.zeros_like(timestamps_interp)
                dh_interp = np.zeros_like(timestamps_interp)
                dw_interp = np.zeros_like(timestamps_interp)
                for interp_id, interp_time in enumerate(timestamps_interp):
                    window_id_start = np.where(timestamp_welding >= interp_time-1/sample_rate)[0][0]
                    window_id_end = np.where(timestamp_welding <= interp_time)[0][-1]+1
                    if window_id_end<=window_id_start:
                        print('Error: window_id_end <= window_id_start, check the data')
                        continue
                    if window_id_end>len(timestamp_welding) or window_id_start>=len(timestamp_welding):
                        print('Error: window_id_end > len(timestamps) or window_id_start >= len(timestamps), check the data')
                        continue
                    cmd_v_interp[interp_id] = np.mean(profile_welding[window_id_start:window_id_end, 2])
                    cmd_fd_interp[interp_id] = np.mean(profile_welding[window_id_start:window_id_end, 3])
                    dh_interp[interp_id] = np.mean(profile_welding[window_id_start:window_id_end, 5])
                    dw_interp[interp_id] = np.mean(profile_welding[window_id_start:window_id_end, 7])
                if np.any(cmd_v_interp==0):
                    # plt.plot(timestamps_interp, cmd_v_interp, 'o', label='cmd_v_interp')
                    # plt.grid()
                    # plt.show()
                    # interpolate the zero values using linear interpolation
                    cmd_v_interp = np.interp(timestamps_interp, timestamps_interp[cmd_v_interp!=0], cmd_v_interp[cmd_v_interp!=0])
                    cmd_fd_interp = np.interp(timestamps_interp, timestamps_interp[cmd_fd_interp!=0], cmd_fd_interp[cmd_fd_interp!=0])
                    dh_interp = np.interp(timestamps_interp, timestamps_interp[dh_interp!=0], dh_interp[dh_interp!=0])
                    dw_interp = np.interp(timestamps_interp, timestamps_interp[dw_interp!=0], dw_interp[dw_interp!=0])

                # save the interpolated data
                interp_data = np.column_stack((timestamps_interp, cmd_v_interp, cmd_fd_interp, dh_interp, dw_interp))
                np.savetxt(this_layer_dir+'profile_welding_'+str(sample_rate)+'_dhdw.csv', interp_data, delimiter=',', header='timestamp,cmd_v,cmd_fd,dh,dw')
                train_data.append(interp_data)
                train_data_batch_len.append(len(interp_data))
                    
    # total amount of data
    print("Total amount of data: ", len(train_data))
    print("Total amount of data batch: ", np.sum(train_data_batch_len))
    # spread the data randomly into 5 totes but with almost equal amount of data
    # spread_epsilon = 0.8
    train_data_split = [[] for _ in range(5)]
    train_data_split_len = np.array([0 for _ in range(5)])
    for dir_i, data_dir in enumerate(data_dirs):
        if np.any(train_data_split_len==0):
            # equally distribute the data to the totes with zero length
            zero_totes = np.where(train_data_split_len==0)[0]
            chosen_tote = np.random.choice(zero_totes)
        else:
            # randomly choose a tote based on the inverse of the current length of the tote
            choice_p = 1/train_data_split_len
            choice_p /= np.sum(choice_p)  # normalize to sum to 1
            chosen_tote = np.random.choice(np.arange(5), p=choice_p)
        train_data_split[chosen_tote].append(train_data[dir_i])
        train_data_split_len[chosen_tote] += train_data_batch_len[dir_i]
    print("Total amount of data in each tote: ", train_data_split_len)
    print("min ratio:", np.min(train_data_split_len)/np.sum(train_data_split_len))
    print("max ratio:", np.max(train_data_split_len)/np.sum(train_data_split_len))

    # training loop
    train