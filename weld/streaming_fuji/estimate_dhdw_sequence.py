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
from model_train_utils import *

np.random.seed(42) # for reproducibility
torch.manual_seed(42) # for reproducibility

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(train_data_input:torch.tensor, train_data_labels:torch.tensor, test_data_input:torch.tensor, test_data_labels:torch.tensor, model:nn.Module, epochs, learning_rate):
    
    # loss function
    loss_fn = nn.MSELoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    training_losses = []
    testing_losses = []
    for epoch in range(epochs):
        # get the testing loss
        model.eval()
        with torch.no_grad():
            test_predictions = model(test_data_labels, test_data_input)
            test_loss = loss_fn(test_predictions, test_data_labels)
            testing_losses.append(test_loss.item())
        
        # save best testing loss model
        if epoch == 0 or test_loss.item() < min(testing_losses):
            best_model = model.state_dict()
            torch.save(best_model, 'weld_LSTM_models/best_model.pth')

        # train the model
        model.train()
        optimizer.zero_grad()

        predictions = model(train_data_labels, train_data_input)
        loss = loss_fn(predictions, train_data_labels)
        training_losses.append(loss.item())

        # print training progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Training Loss: {loss.item():.4f}, Testing Loss: {test_loss.item():.4f}")

        # backpropagation
        loss.backward()
        optimizer.step()

    return model, training_losses, testing_losses

if __name__ == "__main__":

    # load data
    geo_data_dir = '../../data/wall_weld_test/'
    logdata_dir_all = ['weld_fujiscan_2025_06_11_16_27_41/','weld_fujiscan_2025_06_11_16_52_36/','weld_fujiscan_2025_06_11_17_16_48/',\
                       'weld_fujiscan_2025_06_11_17_49_27/','weld_fujiscan_2025_06_11_18_14_56/','weld_fujiscan_2025_06_12_17_33_24/',\
                       'weld_fujiscan_2025_06_12_16_59_09/','weld_fujiscan_2025_06_12_15_33_03/','weld_fujiscan_2025_06_12_15_03_27/']

    # parameters
    sample_rate = 10 # Hz, using the rate of ir camera
    train_test_split = 0.8 # 80% for training, 20% for testing
    epochs = 1000 # number of epochs for training
    sequence_length = 40 # sequence length for training
    sample_sequence_overlap = 0.5 # overlap between sequences, 0.5 means 50% overlap
    learning_rate = 0.001 # learning rate for training

    # model parameters
    model_input_size = 4 # cmd_v, cmd_fd, dh, dw
    # model_input_size = 2 # cmd_v, cmd_fd
    model_hidden_size = 64 # hidden size of the LSTM
    lstm_num_layers = 1 # number of layers in the LSTM
    model_output_size = 2 # dh, dw

    # model initialization
    if model_input_size == 4:
        model = LSTMAutoRegressionModel(input_size=model_input_size, hidden_size=model_hidden_size, output_size=model_output_size, num_layers=lstm_num_layers, device=device).to(device)
    elif model_input_size == 2:
        model = LSTMModel(input_size=model_input_size, hidden_size=model_hidden_size, output_size=model_output_size, num_layers=lstm_num_layers, device=device).to(device)
    else:
        raise ValueError("model_input_size must be 2 or 4")

    ignore_start_end = 5
    start_x = -55 + ignore_start_end
    end_x = 55 - ignore_start_end

    ### data processing
    data_dirs = []
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
                this_layer = np.loadtxt(this_layer_dir+'profile_welding_'+str(sample_rate)+'_dhdw.csv', delimiter=',', skiprows=1)
                train_data_batch_len.append(len(this_layer))
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
                train_data_batch_len.append(len(interp_data))
                    
    # total amount of data
    print("Total amount of data: ", len(data_dirs))
    print("Total amount of data batch: ", np.sum(train_data_batch_len))
    # spread the data randomly into 5 totes but with almost equal amount of data
    # spread_epsilon = 0.8
    train_data_split_dir = [[] for _ in range(5)]
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
        train_data_split_dir[chosen_tote].append(data_dir)
        train_data_split_len[chosen_tote] += train_data_batch_len[dir_i]
    print("Total amount of data in each tote: ", train_data_split_len)
    print("min ratio:", np.min(train_data_split_len)/np.sum(train_data_split_len))
    print("max ratio:", np.max(train_data_split_len)/np.sum(train_data_split_len))

    ###### load data ######
    def load_data_from_tote(data_dir_tote):
        data_all = []
        for dir_tote in data_dir_tote:
            for dir_name in dir_tote:
                this_layer = np.loadtxt(dir_name+'profile_welding_'+str(sample_rate)+'_dhdw.csv', delimiter=',', skiprows=1)
                for in_layer_id in range(0,len(this_layer)-sequence_length, int(sequence_length*(1-sample_sequence_overlap))):
                    if in_layer_id+sequence_length >= len(this_layer):
                        continue
                    data_all.append(this_layer[in_layer_id:in_layer_id+sequence_length, :])
                if np.all(data_all[-1]!=this_layer[-sequence_length:, :]):
                    data_all.append(this_layer[-sequence_length:, :])
        return np.array(data_all)

    train_data_dir_tote = train_data_split_dir[:-1]
    test_data_dir_tote = train_data_split_dir[-1:]
    train_data = load_data_from_tote(train_data_dir_tote)
    test_data = load_data_from_tote(test_data_dir_tote)
    print("Train data shape: ", train_data.shape, "Total samples:", train_data.shape[0]* train_data.shape[1])
    print("Test data shape: ", test_data.shape, "Total samples:", test_data.shape[0]* test_data.shape[1])

    # normalization 
    max_feedrate = np.max(np.append(train_data[:, :, 2], test_data[:, :, 2]))
    min_feedrate = np.min(np.append(train_data[:, :, 2], test_data[:, :, 2]))
    max_v = np.max(np.append(train_data[:, :, 1], test_data[:, :, 1]))
    min_v = np.min(np.append(train_data[:, :, 1], test_data[:, :, 1]))
    train_data[:, :, 1] = (train_data[:, :, 1] - min_v) / (max_v - min_v)
    train_data[:, :, 2] = (train_data[:, :, 2] - min_feedrate) / (max_feedrate - min_feedrate)
    test_data[:, :, 1] = (test_data[:, :, 1] - min_v) / (max_v - min_v)
    test_data[:, :, 2] = (test_data[:, :, 2] - min_feedrate) / (max_feedrate - min_feedrate)

    # prepare data for training
    train_data_input = torch.tensor(train_data[:, :-1, 1:3], dtype=torch.float32).to(device)  # cmd_v, cmd_fd
    train_data_labels = torch.tensor(train_data[:, 1:, 3:5], dtype=torch.float32).to(device)  # dh, dw
    test_data_input = torch.tensor(test_data[:, :-1, 1:3], dtype=torch.float32).to(device)  # cmd_v, cmd_fd
    test_data_labels = torch.tensor(test_data[:, 1:, 3:5], dtype=torch.float32).to(device)  # dh, dw

    # training loop
    model, training_loss, testing_loss = train(train_data_input, train_data_labels, test_data_input, test_data_labels, model, epochs, learning_rate)

    # plot training and testing loss
    plt.figure(figsize=(10, 5))
    plt.plot(training_loss, label='Training Loss')
    plt.plot(testing_loss, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.title('Training and Testing Loss')
    plt.show()

    # plot dh and w error distribution
    model.eval()
    with torch.no_grad():
        train_predictions = model(train_data_labels, train_data_input)
        test_predictions = model(test_data_labels, test_data_input)
        train_dh_error = np.abs((train_predictions[:, :, 0] - train_data_labels[:, :, 0]).cpu().numpy().flatten())
        train_dw_error = np.abs((train_predictions[:, :, 1] - train_data_labels[:, :, 1]).cpu().numpy().flatten())
        test_dh_error = np.abs((test_predictions[:, :, 0] - test_data_labels[:, :, 0]).cpu().numpy().flatten())
        test_dw_error = np.abs((test_predictions[:, :, 1] - test_data_labels[:, :, 1]).cpu().numpy().flatten())
    # dh_error = np.concatenate((train_dh_error, test_dh_error))
    # dw_error = np.concatenate((train_dw_error, test_dw_error))
    plot_error_distribution(train_dh_error, train_dw_error, test_dh_error, test_dw_error)
