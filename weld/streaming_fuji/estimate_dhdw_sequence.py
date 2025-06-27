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

# for plotting
xy_label_size = 14
xy_tick_size = 12
legend_size = 12
title_size = 16
sup_title_size = 18

np.random.seed(42) # for reproducibility
torch.manual_seed(42) # for reproducibility

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

def train(train_data_input:torch.tensor, train_data_labels:torch.tensor, test_data_input:torch.tensor, test_data_labels:torch.tensor, model:nn.Module, \
          history_length, epochs, learning_rate, model_dir='weld_LSTM_models/'):
    
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
            test_loss = loss_fn(test_predictions, test_data_labels[:, history_length:, :]) # skip the history length for loss calculation
            testing_losses.append(test_loss.item())
        
        # save best testing loss model
        if epoch == 0 or test_loss.item() < min(testing_losses):
            best_model = model.state_dict()
            torch.save(best_model, model_dir+'best_model.pth')

        # train the model
        model.train()
        optimizer.zero_grad()

        predictions = model(train_data_labels, train_data_input)
        loss = loss_fn(predictions, train_data_labels[:, history_length:, :])  # skip the history length for loss calculation
        training_losses.append(loss.item())

        # print training progress
        if epoch % (epochs//10) == 0:
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
    
    
    
    train_flag = True # set to False to use the pre-trained model
    model_dir = 'weld_Seq_models/' # directory to save the model
    # model directory
    if train_flag:

        # parameters
        model_type = 'NARMA' # 'LSTM', 'RNN', 'GRU', 'NARMA', 'DTRNN'
        sample_rate = 10 # Hz, using the rate of ir camera
        train_test_split = 0.8 # 80% for training, 20% for testing
        epochs = 5000 # number of epochs for training
        sequence_length = 40 # sequence length for training
        sample_sequence_overlap = 0.5 # overlap between sequences, 0.5 means 50% overlap
        learning_rate = 0.001 # learning rate for training

        # model parameters)
        # model_input_size = 18 # (cmd_v, cmd_fd)_(t,t-1,t-2), (dh,dw)_(t-1,t-2,t-3), (dh dw error)_(t-1,t-2,t-3)
        model_input_size = 12 # (cmd_v, cmd_fd)_(t,t-1,t-2), (dh,dw)_(t-1,t-2,t-3), (dh dw error)_(t-1,t-2,t-3)
        # model_input_size = 4 # cmd_v, cmd_fd, dh error, dw error
        # model_input_size = 2 # cmd_v, cmd_fd
        model_hidden_size = 3 # hidden size
        num_layers = 1 # number of layers
        model_output_size = 2 # dh, dw
        open_loop = True

        # pass system arguments
        # the first argument is model type, the second argument is model_input_size
        for i in range(len(sys.argv)):
            if i == 0:
                continue
            if i == 1:
                model_type = sys.argv[1]
                if model_type not in ['LSTM', 'RNN', 'GRU', 'NARMA', 'DTRNN']:
                    print("Invalid model type. Please choose from 'LSTM', 'RNN', 'GRU', or 'NARMA'.")
                    sys.exit(1)
            if i == 2:
                model_input_size = int(sys.argv[2])
                if model_input_size < 2:
                    print("Invalid model input size. Please provide a value greater than or equal to 2.")
                    sys.exit(1)
                if model_type == 'NARMA':
                    model_input_size = (model_input_size+2)*3
            if i == 3:
                model_hidden_size = int(sys.argv[3])
                if model_hidden_size < 1:
                    print("Invalid model hidden size. Please provide a value greater than or equal to 1.")
                    sys.exit(1)
            if i == 4:
                open_loop = sys.argv[4].lower() == 'true'

        # how many previous time steps to consider, only used for AutoRegression
        if model_type!= 'NARMA':
            history_length = max(0,int(model_input_size/4-0.5))
            open_loop = True if model_input_size == 2 else False # if model_input_size is 2, then it is an open loop model (RNN, LSTM, GRU)
        else:
            if open_loop:
                history_length = max(0,int(model_input_size/4)) 
            else:
                history_length = max(0,int(model_input_size/6))

        if model_type == 'NARMA':
            model_hidden_size = [model_hidden_size,model_hidden_size] # NARMA model hidden size is a list of two elements, [first hidden, second hidden]
        if model_type == 'DTRNN':
            num_layers = 2

        training_params = {
            'geo_data_dir': geo_data_dir, 'logdata_dir_all': logdata_dir_all,
            'model_type': model_type,
            'sample_rate': sample_rate, 'train_test_split': train_test_split, 'epochs': epochs,
            'sequence_length': sequence_length, 'sample_sequence_overlap': sample_sequence_overlap,
            'learning_rate': learning_rate, 'model_input_size': model_input_size,
            'history_length': history_length, 'model_hidden_size': model_hidden_size,
            'num_layers': num_layers, 'model_output_size': model_output_size,
            'open_loop': open_loop
        }
        # save the training parameters
        # add timestamp to the model_dir
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        model_dir = model_dir + "model_"+ timestamp + '/'
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
        with open(model_dir+'training_params.yaml', 'w') as f:
            yaml.dump(training_params, f, default_flow_style=False)
    else:
        model_dir = model_dir+ 'model_20250513_153408/'
        # load the training parameters
        with open(model_dir+'training_params.yaml', 'r') as f:
            training_params = yaml.safe_load(f)
        geo_data_dir = training_params['geo_data_dir']
        logdata_dir_all = training_params['logdata_dir_all']
        model_type = training_params['model_type']
        sample_rate = training_params['sample_rate']
        train_test_split = training_params['train_test_split']
        epochs = training_params['epochs']
        sequence_length = training_params['sequence_length']
        sample_sequence_overlap = training_params['sample_sequence_overlap']
        learning_rate = training_params['learning_rate']
        model_input_size = training_params['model_input_size']
        history_length = training_params['history_length']
        model_hidden_size = training_params['model_hidden_size']
        num_layers = training_params['num_layers']
        model_output_size = training_params['model_output_size']
        open_loop = training_params['open_loop']

    print("=============================================")
    print("Training parameters:")
    print("Model type:", model_type, "Model input size:", model_input_size, "Model hidden size:", model_hidden_size)

    # Model types
    if model_type == 'LSTM':
        if model_input_size == 2:
            modelClass = LSTMModel
        else:
            modelClass = LSTMAutoRegressionModel
    elif model_type == 'RNN' or model_type == 'DTRNN':
        if model_input_size == 2 and model_type == 'RNN':
            modelClass = RNNModel
        else:
            modelClass = RNNAutoRegressionModel
    elif model_type == 'GRU':
        if model_input_size == 2:
            modelClass = GRUModel
        else:
            modelClass = GRUAutoRegressionModel
    elif model_type == 'NARMA':
        modelClass = ARMANeuralNetwork

    # model initialization
    if model_input_size == 2 and model_type not in ['NARMA', 'DTRNN']:
        model = modelClass(input_size=model_input_size, hidden_size=model_hidden_size, output_size=model_output_size, num_layers=num_layers, device=device).to(device)
    else:
        model = modelClass(input_size=model_input_size, hidden_size=model_hidden_size, output_size=model_output_size, num_layers=num_layers, history_length=history_length, open_loop=open_loop, device=device).to(device)
    print("Model trainable parameters:",count_parameters(model))
    # exit()

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
    train_data_input = torch.tensor(train_data[:, :, 1:3], dtype=torch.float32).to(device)  # cmd_v, cmd_fd
    train_data_labels = torch.tensor(train_data[:, :, 3:5], dtype=torch.float32).to(device)  # dh, dw
    test_data_input = torch.tensor(test_data[:, :, 1:3], dtype=torch.float32).to(device)  # cmd_v, cmd_fd
    test_data_labels = torch.tensor(test_data[:, :, 3:5], dtype=torch.float32).to(device)  # dh, dw

    # padd history length at the beginning of the input data
    if history_length > 0:
        # repeat the first input for history length times and pad
        print("Padding history length: ", history_length)
        train_data_input = torch.cat((train_data_input[:,0:1,:].repeat(1,history_length,1), train_data_input), dim=1).to(device)
        test_data_input = torch.cat((test_data_input[:,0:1,:].repeat(1,history_length,1), test_data_input), dim=1).to(device)
        # padd zeros to the labels
        train_data_labels = torch.cat((torch.zeros((train_data_labels.shape[0], history_length, train_data_labels.shape[2]), dtype=torch.float32).to(device), train_data_labels), dim=1)
        test_data_labels = torch.cat((torch.zeros((test_data_labels.shape[0], history_length, test_data_labels.shape[2]), dtype=torch.float32).to(device), test_data_labels), dim=1)

    print("Train data input shape: ", train_data_input.shape)
    print("Train data labels shape: ", train_data_labels.shape)
    
    if train_flag:
        # training loop
        model, training_loss, testing_loss = train(train_data_input, train_data_labels, test_data_input, test_data_labels, model,\
                                               history_length, epochs, learning_rate, model_dir=model_dir)
        # save loss
        np.savetxt(model_dir+'training_loss.csv', training_loss, delimiter=',')
        np.savetxt(model_dir+'testing_loss.csv', testing_loss, delimiter=',')
    else:
        # load the pre-trained model
        model.load_state_dict(torch.load(model_dir+'best_model.pth'))
        print("Loaded pre-trained model from: ", model_dir+'best_model.pth')
        training_loss = np.loadtxt(model_dir+'training_loss.csv', delimiter=',')
        testing_loss = np.loadtxt(model_dir+'testing_loss.csv', delimiter=',')

    # plot training and testing loss
    plt.figure(figsize=(10, 5))
    plt.plot(training_loss, label='Training Loss')
    plt.plot(testing_loss, label='Testing Loss')
    plt.xlabel('Epochs', fontsize=xy_label_size)
    plt.ylabel('Loss', fontsize=xy_label_size)
    plt.xticks(fontsize=xy_tick_size)
    plt.yticks(fontsize=xy_tick_size)
    plt.grid()
    plt.legend(fontsize=legend_size)
    plt.title('Training and Testing Loss', fontsize=title_size)
    plt.tight_layout()
    plt.savefig(model_dir+'training_testing_loss.png')
    # plt.show()

    # plot dh and w error distribution
    model.eval()
    with torch.no_grad():
        train_predictions = model(train_data_labels, train_data_input)
        test_predictions = model(test_data_labels, test_data_input)
        train_dh_error = np.abs((train_predictions[:, :, 0] - train_data_labels[:, history_length:, 0]).cpu().numpy().flatten())
        train_dw_error = np.abs((train_predictions[:, :, 1] - train_data_labels[:, history_length:, 1]).cpu().numpy().flatten())
        test_dh_error = np.abs((test_predictions[:, :, 0] - test_data_labels[:, history_length:, 0]).cpu().numpy().flatten())
        test_dw_error = np.abs((test_predictions[:, :, 1] - test_data_labels[:, history_length:, 1]).cpu().numpy().flatten())
    # dh_error = np.concatenate((train_dh_error, test_dh_error))
    # dw_error = np.concatenate((train_dw_error, test_dw_error))
    plot_error_distribution(train_dh_error, train_dw_error, test_dh_error, test_dw_error,save_dir=model_dir)
