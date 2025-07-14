import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import sys, datetime, yaml, pathlib, glob, os, time
sys.path.append('../../mocap/')
from Models import *
from model_train_utils import *
from qpsolvers import solve_qp

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

# Define hook to freeze half of W_ih
def freeze_half_weight(param: torch.Tensor, freeze_cols=[0,1]):
    mask = torch.zeros_like(param)
    mask[:, freeze_cols] = 1.0  # Freeze left half
    def hook(grad):
        return grad * (1 - mask)
    param.register_hook(hook)

def train(train_data_input:torch.tensor, train_data_labels:torch.tensor, test_data_input:torch.tensor, test_data_labels:torch.tensor, model:nn.Module, \
          history_length, epochs, learning_rate, model_dir='weld_LSTM_models/'):
    
    # loss function
    loss_fn = nn.MSELoss()
    # optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

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
        if epoch == 0 or test_loss.item() <= min(testing_losses):
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
    load_pretrained = True
    
    if len(sys.argv) > 1:
        train_flag = True if sys.argv[1].lower() == 'true' else False  # first argument is train flag, if not provided, default to True

    model_dir = 'weld_Seq_models/' # directory to save the model
    # model directory
    if train_flag and not load_pretrained:

        # parameters
        model_type = 'RNN' # 'LSTM', 'RNN', 'GRU', 'NARMA', 'DTRNN'
        sample_rate = 10 # Hz, using the rate of ir camera
        train_test_split = 0.8 # 80% for training, 20% for testing
        epochs = 5000 # number of epochs for training
        sequence_length = 40 # sequence length for training
        sample_sequence_overlap = 0.5 # overlap between sequences, 0.5 means 50% overlap
        learning_rate = 0.001 # learning rate for training
        # latency = 1

        # model parameters
        # model_input_size = 18 # (cmd_v, cmd_fd)_(t,t-1,t-2), (dh,dw)_(t-1,t-2,t-3), (dh dw error)_(t-1,t-2,t-3)
        # model_input_size = 12 # (cmd_v, cmd_fd)_(t,t-1,t-2), (dh,dw)_(t-1,t-2,t-3), (dh dw error)_(t-1,t-2,t-3)
        model_input_size = 4 # cmd_v, cmd_fd, dh error, dw error
        # model_input_size = 2 # cmd_v, cmd_fd
        # model_input_size = 5 # cmd_v, cmd_fd, stickout, dh error, dw error
        # model_input_size = 3 # cmd_v, cmd_fd, stickout
        use_stickout_length = True if model_input_size in [3,5] else False

        model_hidden_size = 3 # hidden size
        num_layers = 1 # number of layers
        model_output_size = 2 # dh, dw
        open_loop = False

        # pass system arguments
        # the first argument is model type, the second argument is model_input_size
        for i in range(len(sys.argv)):
            if i < 2:
                continue
            if i == 2:
                model_type = sys.argv[1]
                if model_type not in ['LSTM', 'RNN', 'GRU', 'NARMA', 'DTRNN']:
                    print("Invalid model type. Please choose from 'LSTM', 'RNN', 'GRU', or 'NARMA'.")
                    sys.exit(1)
            if i == 3:
                model_input_size = int(sys.argv[2])
                if model_input_size < 2:
                    print("Invalid model input size. Please provide a value greater than or equal to 2.")
                    sys.exit(1)
                if model_type == 'NARMA':
                    model_input_size = (model_input_size+2)*3
            if i == 4:
                model_hidden_size = int(sys.argv[3])
                if model_hidden_size < 1:
                    print("Invalid model hidden size. Please provide a value greater than or equal to 1.")
                    sys.exit(1)
            if i == 5:
                open_loop = sys.argv[4].lower() == 'true'

        # how many previous time steps to consider, only used for AutoRegression
        if model_type!= 'NARMA':
            history_length = max(0,int(model_input_size/4-0.5))
            open_loop = True if model_input_size in [2,3] else False # if model_input_size is 2, then it is an open loop model (RNN, LSTM, GRU)
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
            'open_loop': open_loop, 'use_stickout_length': use_stickout_length
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
        # RNN 8 hidden close/open: 20250625_131753/20250625_131507
        # RNN 16 hidden close/open: 20250625_132020/20250625_131520
        pre_trained_model_dir = model_dir+'model_20250625_131507/'
        if len(sys.argv) < 2:
            model_dir = deepcopy(pre_trained_model_dir)
        else:
            model_dir = model_dir + sys.argv[2] + '/'
                    
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
        num_layers = training_params['num_layers'] if 'num_layers' in training_params else 1
        model_output_size = training_params['model_output_size']
        if 'open_loop' in training_params:
            open_loop = training_params['open_loop']
        else:
            open_loop = True if model_input_size == 2 else False
        
        if train_flag:
            try:
                use_stickout_length = training_params['use_stickout_length']
            except KeyError:
                use_stickout_length = False
            # using the pre-trained model directory to train a new model
            model_input_size = 5 if use_stickout_length else 4
            if model_input_size > 3:
                open_loop = False
            # epochs = 100 # for testing purpose, reduce the epochs to 100
            epochs = 1000 # only 1000 epochs for training with pre-trained model

            training_params['model_input_size'] = model_input_size
            training_params['open_loop'] = open_loop
            training_params['epochs'] = epochs
            training_params['pre_trained_model_dir'] = pre_trained_model_dir

            # save the training parameters
            # add timestamp to the model_dir
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            model_dir = "weld_Seq_models/model_"+ timestamp + '/'
            pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
            with open(model_dir+'training_params.yaml', 'w') as f:
                yaml.dump(training_params, f, default_flow_style=False)

    print("Training parameters:")
    print("Train flag:", train_flag)
    print("Model directory:", model_dir)
    if (not train_flag) or load_pretrained:
        print("Using pre-trained model:", pre_trained_model_dir)
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
                stickout_interp = np.zeros_like(timestamps_interp)
                thermal_interp = np.zeros_like(timestamps_interp)
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
                    stickout_interp[interp_id] = np.mean(profile_welding[window_id_start:window_id_end, 6])
                    thermal_interp[interp_id] = np.mean(profile_welding[window_id_start:window_id_end, 9])
                if np.any(cmd_v_interp==0):
                    # plt.plot(timestamps_interp, cmd_v_interp, 'o', label='cmd_v_interp')
                    # plt.grid()
                    # plt.show()
                    # interpolate the zero values using linear interpolation
                    cmd_v_interp = np.interp(timestamps_interp, timestamps_interp[cmd_v_interp!=0], cmd_v_interp[cmd_v_interp!=0])
                    cmd_fd_interp = np.interp(timestamps_interp, timestamps_interp[cmd_fd_interp!=0], cmd_fd_interp[cmd_fd_interp!=0])
                    dh_interp = np.interp(timestamps_interp, timestamps_interp[dh_interp!=0], dh_interp[dh_interp!=0])
                    dw_interp = np.interp(timestamps_interp, timestamps_interp[dw_interp!=0], dw_interp[dw_interp!=0])
                    stickout_interp = np.interp(timestamps_interp, timestamps_interp[stickout_interp!=0], stickout_interp[stickout_interp!=0])
                    thermal_interp = np.interp(timestamps_interp, timestamps_interp[thermal_interp!=0], thermal_interp[thermal_interp!=0])

                # save the interpolated data
                interp_data = np.column_stack((timestamps_interp, cmd_v_interp, cmd_fd_interp, dh_interp, dw_interp, stickout_interp, thermal_interp))
                np.savetxt(this_layer_dir+'profile_welding_'+str(sample_rate)+'_dhdw.csv', interp_data, delimiter=',', header='timestamp,cmd_v,cmd_fd,dh,dw,stickout,thermal')
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

    # save input cmd_v cmd_feedrate
    train_cmd_v = train_data[:, :, 1].flatten()
    train_cmd_feedrate = train_data[:, :, 2].flatten()
    train_stickout_length = train_data[:, :, 5].flatten()
    test_cmd_v = test_data[:, :, 1].flatten()
    test_cmd_feedrate = test_data[:, :, 2].flatten()
    test_stickout_length = test_data[:, :, 5].flatten()

    np.savetxt(model_dir+'../train_cmd_v_feedrate.csv', np.vstack((train_cmd_v, train_cmd_feedrate, train_stickout_length)).T, delimiter=',', header='cmd_v,cmd_fd,stickout')
    np.savetxt(model_dir+'../test_cmd_v_feedrate.csv', np.vstack((test_cmd_v, test_cmd_feedrate, test_stickout_length)).T, delimiter=',', header='cmd_v,cmd_fd,stickout')

    train_data[:, :, 1] = (train_data[:, :, 1] - min_v) / (max_v - min_v)
    train_data[:, :, 2] = (train_data[:, :, 2] - min_feedrate) / (max_feedrate - min_feedrate)
    test_data[:, :, 1] = (test_data[:, :, 1] - min_v) / (max_v - min_v)
    test_data[:, :, 2] = (test_data[:, :, 2] - min_feedrate) / (max_feedrate - min_feedrate)

    # prepare data for training
    if use_stickout_length:
        train_data_input = torch.tensor(train_data[:, :, [1,2,5]], dtype=torch.float32).to(device)  # cmd_v, cmd_fd, stickout
        test_data_input = torch.tensor(test_data[:, :, [1,2,5]], dtype=torch.float32).to(device)  # cmd_v, cmd_fd, stickout
    else:
        train_data_input = torch.tensor(train_data[:, :, 1:3], dtype=torch.float32).to(device)  # cmd_v, cmd_fd
        test_data_input = torch.tensor(test_data[:, :, 1:3], dtype=torch.float32).to(device)  # cmd_v, cmd_fd
    train_data_labels = torch.tensor(train_data[:, :, 3:5], dtype=torch.float32).to(device)  # dh, dw
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
        if load_pretrained:
            print("Loading pre-trained model from: ", pre_trained_model_dir+'best_model.pth')
            parameter_dict = torch.load(pre_trained_model_dir+'best_model.pth', weights_only=True)
            try:
                model.load_state_dict(parameter_dict)
            except RuntimeError as e:
                print("Train closed loop from open loop pre-trained model.")
                with torch.no_grad():
                    model.rnn_cell.weight_hh.data.copy_(parameter_dict['rnn.weight_hh_l0'])
                    model.rnn_cell.bias_hh.data.copy_(parameter_dict['rnn.bias_hh_l0'])
                    model.rnn_cell.weight_ih[:, :2].data.copy_(parameter_dict['rnn.weight_ih_l0'])
                    model.rnn_cell.bias_ih.data.copy_(parameter_dict['rnn.bias_ih_l0'])
                    model.fc.weight.data.copy_(parameter_dict['fc.weight'])
                    model.rnn_cell.weight_hh.requires_grad = False
                    model.rnn_cell.bias_hh.requires_grad = False
                    freeze_half_weight(model.rnn_cell.weight_ih, freeze_cols=[0,1]) # freeze the first two columns of weight_ih
                    model.rnn_cell.bias_ih.requires_grad = False
                    model.fc.weight.requires_grad = False
                    pre_train_weight_hh = model.rnn_cell.weight_hh.detach().cpu().numpy()
                    pre_train_weight_ih = model.rnn_cell.weight_ih.detach().cpu().numpy()
        # training loop
        start_time = time.time()
        _, training_loss, testing_loss = train(train_data_input, train_data_labels, test_data_input, test_data_labels, model,\
                                                history_length, epochs, learning_rate, model_dir=model_dir)
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")

        # plot pre-trained and after training model parameters differences
        # if load_pretrained:
        #     plt.matshow(pre_train_weight_hh-model.rnn_cell.weight_hh.detach().cpu().numpy(), cmap='viridis', aspect='equal')
        #     plt.title("Weight HH Difference")
        #     plt.colorbar()
        #     plt.show()
        #     plt.matshow(pre_train_weight_ih-model.rnn_cell.weight_ih.detach().cpu().numpy(), cmap='viridis', aspect='equal')
        #     plt.title("Weight IH Difference")
        #     plt.colorbar()
        #     plt.show()

        model.load_state_dict(torch.load(model_dir+'best_model.pth',weights_only=True)) # load the best model for evaluation
        # save loss
        np.savetxt(model_dir+'training_loss.csv', training_loss, delimiter=',')
        np.savetxt(model_dir+'testing_loss.csv', testing_loss, delimiter=',')
        np.savetxt(model_dir+'training_time.csv', np.array([end_time - start_time]), delimiter=',')
    else:
        # load the pre-trained model
        model.load_state_dict(torch.load(model_dir+'best_model.pth',weights_only=True))

        # test_u = np.zeros((1,1000,2))
        # test_u = torch.tensor(test_u, dtype=torch.float32).to(device)
        # test_x = np.ones((1,1000,2))*2 # 1 sample, 1000 time steps, model_input_size features
        # test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
        # model.eval()
        # with torch.no_grad():
        #     test_y = model(test_x, test_u)
        # test_y = test_y.cpu().numpy().astype(np.float64)
        # plt.plot(test_y[0,:,0], label='dh prediction')
        # plt.plot(test_y[0,:,1], label='dw prediction')
        # plt.xlabel('Time Step', fontsize=xy_label_size)
        # plt.ylabel('Prediction', fontsize=xy_label_size)
        # plt.title('Model Prediction', fontsize=title_size)
        # plt.legend(fontsize=legend_size)
        # plt.show()

        print("Loaded pre-trained model from: ", model_dir+'best_model.pth')
        training_loss = np.loadtxt(model_dir+'training_loss.csv', delimiter=',')
        testing_loss = np.loadtxt(model_dir+'testing_loss.csv', delimiter=',')

        # visualize the parameters
        if model_type == 'RNN':
            open_loop_string = 'Open Loop' if open_loop else 'Closed Loop'
            if open_loop:
                Whh = model.rnn.weight_hh_l0.detach().cpu().numpy().astype(np.float64)
                Wih = model.rnn.weight_ih_l0.detach().cpu().numpy().astype(np.float64)
                bh = model.rnn.bias_hh_l0.detach().cpu().numpy().astype(np.float64)
                bi = model.rnn.bias_ih_l0.detach().cpu().numpy().astype(np.float64)
            else:
                Whh = model.rnn_cell.weight_hh.detach().cpu().numpy().astype(np.float64)
                Wih = model.rnn_cell.weight_ih.detach().cpu().numpy().astype(np.float64)
                bh = model.rnn_cell.bias_hh.detach().cpu().numpy().astype(np.float64)
                bi = model.rnn_cell.bias_ih.detach().cpu().numpy().astype(np.float64)
            Woh = model.fc.weight.detach().cpu().numpy().astype(np.float64)
            bo = model.fc.bias.detach().cpu().numpy().astype(np.float64)

            print("Whh shape:", Whh.shape, "Wih shape:", Wih.shape, "bh shape:", bh.shape, "bi shape:", bi.shape)
            print("Woh shape:", Woh.shape, "bo shape:", bo.shape)

            # find the equilibrium point of the RNN

            # find the matrix elimited the parameter redundancy
            W_hh_min = Whh + Wih[:,-2:]@Woh

            if not open_loop:
                prediction,zt = model.forward_linear_mat(test_data_labels, test_data_input)
                prediction_true = model(test_data_labels, test_data_input)

                # make sure prediction and prediction_true are the same
                assert np.allclose(prediction.detach().cpu().numpy(), prediction_true.detach().cpu().numpy()), "Prediction and prediction_true are not the same!"

                zt_flatten = zt.view(-1).detach().cpu().numpy()
                zt_tanh_derivative = 1 - np.tanh(zt_flatten)**2
                # plot zt_tanh_derivative distribution
                plt.figure(figsize=(10, 5))
                plt.hist(zt_tanh_derivative, bins=100, density=True, alpha=0.7, color='blue')
                plt.title('Distribution of $z_t$ Tanh Derivative', fontsize=title_size)
                plt.xlabel('$z_t$ Tanh Derivative', fontsize=xy_label_size)
                plt.ylabel('Density', fontsize=xy_label_size)
                plt.xticks(fontsize=xy_tick_size)
                plt.yticks(fontsize=xy_tick_size)
                plt.grid()
                plt.tight_layout()
                plt.show()

            # plot Whh and Wih
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 4, 1)
            plt.imshow(Whh, cmap='viridis', aspect='equal')
            plt.colorbar()
            plt.title(f'RNN $W_{{hh}}$ Matrix', fontsize=title_size)
            plt.xlabel('Hidden Units', fontsize=xy_label_size)
            plt.ylabel('Hidden Units', fontsize=xy_label_size)
            plt.xticks(fontsize=xy_tick_size)
            plt.yticks(fontsize=xy_tick_size)
            plt.subplot(1, 4, 2)
            plt.imshow(Wih, cmap='viridis', aspect='equal')
            plt.colorbar()
            plt.title(f'RNN $W_{{ih}}$ Matrix', fontsize=title_size)
            plt.xlabel('Input Features', fontsize=xy_label_size)
            plt.ylabel('Hidden Units', fontsize=xy_label_size)
            plt.xticks(fontsize=xy_tick_size)
            plt.yticks(fontsize=xy_tick_size)
            plt.suptitle(f'RNN Weight Matrices {open_loop_string}', fontsize=sup_title_size)
            plt.subplot(1, 4, 3)
            plt.imshow(W_hh_min, cmap='viridis', aspect='equal')
            plt.colorbar()
            plt.title(f'RNN $W_{{hh,min}}$ Matrix', fontsize=title_size)
            plt.xlabel('Hidden Units', fontsize=xy_label_size)
            plt.ylabel('Hidden Units', fontsize=xy_label_size)
            plt.xticks(fontsize=xy_tick_size)
            plt.yticks(fontsize=xy_tick_size)
            # plt.subplot(1, 4, 3)
            # plt.imshow(bh.reshape(-1, 1), cmap='viridis', aspect='equal')
            # plt.colorbar()
            # plt.title(f'RNN $b_{{h}}$ Vector', fontsize=title_size)
            # plt.xlabel('Hidden Units', fontsize=xy_label_size)
            # plt.ylabel('Bias', fontsize=xy_label_size)
            # plt.xticks(fontsize=xy_tick_size)
            # plt.yticks(fontsize=xy_tick_size)
            # plt.subplot(1, 4, 4)
            # plt.imshow(bi.reshape(-1, 1), cmap='viridis', aspect='equal')
            # plt.colorbar()
            # plt.title(f'RNN $b_{{i}}$ Vector', fontsize=title_size)
            # plt.xlabel('Input Features', fontsize=xy_label_size)
            # plt.ylabel('Bias', fontsize=xy_label_size)
            # plt.xticks(fontsize=xy_tick_size)
            # plt.yticks(fontsize=xy_tick_size)
            plt.tight_layout()
            plt.show()

            # eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eig(Whh)
            plt.plot(np.sort(np.abs(eigenvalues))[::-1], 'o')
            plt.title(f'Eigenvalues of RNN $W_{{hh}}$ Matrix ({open_loop_string})', fontsize=title_size)
            plt.xlabel('Index', fontsize=xy_label_size)
            plt.ylabel('Eigenvalue Magnitude', fontsize=xy_label_size)
            plt.xticks(fontsize=xy_tick_size)
            plt.yticks(fontsize=xy_tick_size)
            plt.grid()
            plt.tight_layout()
            plt.show()

            eigenvalues, eigenvectors = np.linalg.eig(W_hh_min)
            plt.plot(np.sort(np.abs(eigenvalues))[::-1], 'o')
            plt.title(f'Eigenvalues of RNN $W_{{hh}}$ Matrix ({open_loop_string})', fontsize=title_size)
            plt.xlabel('Index', fontsize=xy_label_size)
            plt.ylabel('Eigenvalue Magnitude', fontsize=xy_label_size)
            plt.xticks(fontsize=xy_tick_size)
            plt.yticks(fontsize=xy_tick_size)
            plt.grid()
            plt.tight_layout()
            plt.show()

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
        train_dh_error = (train_predictions[:, :, 0] - train_data_labels[:, history_length:, 0]).cpu().numpy().flatten()
        train_dw_error = (train_predictions[:, :, 1] - train_data_labels[:, history_length:, 1]).cpu().numpy().flatten()
        test_dh_error = (test_predictions[:, :, 0] - test_data_labels[:, history_length:, 0]).cpu().numpy().flatten()
        test_dw_error = (test_predictions[:, :, 1] - test_data_labels[:, history_length:, 1]).cpu().numpy().flatten()
    
    # save the errors
    np.savetxt(model_dir+'train_dh_error.csv', train_dh_error, delimiter=',')
    np.savetxt(model_dir+'train_dw_error.csv', train_dw_error, delimiter=',')
    np.savetxt(model_dir+'test_dh_error.csv', test_dh_error, delimiter=',')
    np.savetxt(model_dir+'test_dw_error.csv', test_dw_error, delimiter=',')
    # plot the error distribution
    plot_error_distribution(np.abs(train_dh_error), np.abs(train_dw_error), np.abs(test_dh_error), np.abs(test_dw_error), save_dir=model_dir)
    # print test error statistics
    print(f"Test dh Error: Mean = {np.mean(np.abs(test_dh_error)):.4f}, width Error = {np.mean(np.abs(test_dw_error)):.4f}")

    # if no plotting
    exit()

    # plot test data prediction dh dw vs ground truth dh dw of four sequences, using a 2x2 grid
    layer_dir_chosen = np.random.choice(test_data_dir_tote[0], size=8, replace=False)
    layer_dir_chosen = layer_dir_chosen[[0,3,6,7]]  # choose 4 layers for visualization
    chosen_VPD_feedrate = []
    dh_prediction_gt = []
    dw_prediction_gt = []
    timestamps_layer = []
    for i, dir_name in enumerate(layer_dir_chosen):
        this_layer = np.loadtxt(dir_name+'profile_welding_'+str(sample_rate)+'_dhdw.csv', delimiter=',', skiprows=1)
        with open(dir_name+'../weld_meta_data.yml', 'r') as f:
            meta_data = yaml.safe_load(f)
            this_vpd = meta_data['VPD']
        this_feedrate = this_layer[0, 2]
        chosen_VPD_feedrate.append((this_vpd, this_feedrate))
        gt_labels = this_layer[:, 3:5]  # dh, dw
        control_inputs = this_layer[:, 1:3]  # cmd_v, cmd_fd
        control_inputs[:, 0] = (control_inputs[:, 0] - min_v) / (max_v - min_v)  # normalize cmd_v
        control_inputs[:, 1] = (control_inputs[:, 1] - min_feedrate) / (max_feedrate - min_feedrate)  # normalize cmd_fd
        # to tensor with shape (1, sequence_length, 2)
        gt_labels = torch.tensor(gt_labels, dtype=torch.float32).unsqueeze(0).to(device)
        control_inputs = torch.tensor(control_inputs, dtype=torch.float32).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            # predictions = model(gt_labels, control_inputs)
            predictions_all=[]
            dh_prediction_all = []
            dw_prediction_all = []
            latency_t = 20 # latency in time steps, 20 time steps = 2 seconds
            for step_t in range(0,len(gt_labels[0])):
                print("Processing layer "+str(i+1)+" step "+str(step_t+1)+" of "+str(len(gt_labels[0])-latency_t))
                predictions,_ = model.forward_half_obs(gt_labels[:,:step_t,:], control_inputs[:,:min(step_t+latency_t,len(control_inputs[0])),:])
            # predictions = model(gt_labels, control_inputs)
                predictions = predictions.cpu().numpy().astype(np.float64).squeeze(0)
                predictions_all.append(predictions)
                dh_prediction_all.append(predictions[:, 0])
                dw_prediction_all.append(predictions[:, 1])
            gt_labels = gt_labels.cpu().numpy().astype(np.float64).squeeze(0)
        # dh_prediction_gt.append(np.vstack((predictions[:, 0], gt_labels[:, 0])))
        # dw_prediction_gt.append(np.vstack((predictions[:, 1], gt_labels[:, 1])))
        dh_prediction_gt.append((dh_prediction_all, gt_labels[:, 0]))
        dw_prediction_gt.append((dw_prediction_all, gt_labels[:, 1]))
        timestamps_layer.append(this_layer[:, 0])

    # fig, axs = plt.subplots(2, 2, figsize=(20, 14))
    # for i, dh in enumerate(dh_prediction_gt):
    #     timestep_predict = timestamps_layer[i]-timestamps_layer[i][0]
    #     axs[i//2, i%2].clear()  # clear the axes for each time step
    #     axs[i//2, i%2].plot(timestep_predict, dh[0], label=f'Predicted $\Delta h$', color='tab:blue')
    #     axs[i//2, i%2].plot(timestamps_layer[i]-timestamps_layer[i][0], dh[1], label=f'Ground Truth $\Delta h$', color='tab:orange')
    #     axs[i//2, i%2].set_title(f'Sequence {i+1} - VPD: {round(chosen_VPD_feedrate[i][0])}, Feedrate: {round(chosen_VPD_feedrate[i][1])}', fontsize=title_size)
    #     axs[i//2, i%2].set_xlabel('Time Step', fontsize=xy_label_size)
    #     axs[i//2, i%2].set_ylabel(f'$\Delta h$', fontsize=xy_label_size)
    #     axs[i//2, i%2].tick_params(axis='x', labelsize=xy_tick_size)
    #     axs[i//2, i%2].tick_params(axis='y', labelsize=xy_tick_size)
    #     axs[i//2, i%2].legend(fontsize=legend_size)
    #     axs[i//2, i%2].grid()
    # plt.suptitle(f'Test Data $\Delta h$ Prediction vs Ground Truth, ' + open_loop_string, fontsize=sup_title_size)
    # plt.show()

    # fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # for i, dw in enumerate(dw_prediction_gt):
    #     timestep_predict = timestamps_layer[i]-timestamps_layer[i][0]
    #     axs[i//2, i%2].clear()  # clear the axes for each time step
    #     axs[i//2, i%2].plot(timestep_predict, dw[0], label='Predicted $width$', color='tab:blue')
    #     axs[i//2, i%2].plot(timestep_predict, dw[1], label='Ground Truth $width$', color='tab:orange')
    #     axs[i//2, i%2].set_title(f'Sequence {i+1} - VPD: {round(chosen_VPD_feedrate[i][0])}, Feedrate: {round(chosen_VPD_feedrate[i][1])}', fontsize=title_size)
    #     axs[i//2, i%2].set_xlabel('Time Step', fontsize=xy_label_size)
    #     axs[i//2, i%2].set_ylabel('$width$', fontsize=xy_label_size)
    #     axs[i//2, i%2].tick_params(axis='x', labelsize=xy_tick_size)
    #     axs[i//2, i%2].tick_params(axis='y', labelsize=xy_tick_size)
    #     axs[i//2, i%2].legend(fontsize=legend_size)
    #     axs[i//2, i%2].grid()
    # plt.suptitle(f'Test Data $width$ Prediction vs Ground Truth, ' + open_loop_string, fontsize=sup_title_size)
    # plt.show()

    # find the layer with longest sequence
    max_length = max([len(t) for t in timestamps_layer])

    fig, axs = plt.subplots(2, 2, figsize=(20, 14))
    for t_id in range(max_length):
        for i, dh in enumerate(dh_prediction_gt):
            timestep_predict = timestamps_layer[i]-timestamps_layer[i][0]
            timestep_predict = timestep_predict[:len(dh[0][min(t_id,len(dh[0])-1)])]  # adjust the length to match the prediction
            axs[i//2, i%2].clear()  # clear the axes for each time step
            axs[i//2, i%2].plot(timestep_predict, dh[0][min(t_id,len(dh[0])-1)], label=f'Predicted $\Delta h$', color='tab:blue')
            axs[i//2, i%2].plot(timestamps_layer[i]-timestamps_layer[i][0], dh[1], label=f'Ground Truth $\Delta h$', color='tab:orange')
            axs[i//2, i%2].scatter(timestep_predict[-1],dh[0][min(t_id,len(dh[0])-1)][-1], color='red',s=50)
            axs[i//2, i%2].set_title(f'Sequence {i+1} - VPD: {round(chosen_VPD_feedrate[i][0])}, Feedrate: {round(chosen_VPD_feedrate[i][1])}', fontsize=title_size)
            axs[i//2, i%2].set_xlabel('Time Step', fontsize=xy_label_size)
            axs[i//2, i%2].set_ylabel(f'$\Delta h$', fontsize=xy_label_size)
            axs[i//2, i%2].tick_params(axis='x', labelsize=xy_tick_size)
            axs[i//2, i%2].tick_params(axis='y', labelsize=xy_tick_size)
            axs[i//2, i%2].legend(fontsize=legend_size)
            axs[i//2, i%2].grid()
        plt.suptitle(f'Test Data $\Delta h$ Prediction vs Ground Truth, ' + open_loop_string, fontsize=sup_title_size)
        if t_id < max_length-1:
            plt.pause(0.1)  # pause to visualize the animation
        else:
            plt.show()
        if t_id==0:
            input("Press Enter to continue...")
            time.sleep(3)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for t_id in range(max_length):
        for i, dw in enumerate(dw_prediction_gt):
            timestep_predict = timestamps_layer[i]-timestamps_layer[i][0]
            timestep_predict = timestep_predict[:len(dw[0][min(t_id,len(dw[0])-1)])]  # adjust the length to match the prediction
            axs[i//2, i%2].clear()  # clear the axes for each time step
            axs[i//2, i%2].plot(timestep_predict, dw[0][min(t_id,len(dw[0])-1)], label='Predicted $width$', color='tab:blue')
            axs[i//2, i%2].plot(timestep_predict, dw[1], label='Ground Truth $width$', color='tab:orange')
            axs[i//2, i%2].set_title(f'Sequence {i+1} - VPD: {round(chosen_VPD_feedrate[i][0])}, Feedrate: {round(chosen_VPD_feedrate[i][1])}', fontsize=title_size)
            axs[i//2, i%2].set_xlabel('Time Step', fontsize=xy_label_size)
            axs[i//2, i%2].set_ylabel('$width$', fontsize=xy_label_size)
            axs[i//2, i%2].tick_params(axis='x', labelsize=xy_tick_size)
            axs[i//2, i%2].tick_params(axis='y', labelsize=xy_tick_size)
            axs[i//2, i%2].legend(fontsize=legend_size)
            axs[i//2, i%2].grid()
        plt.suptitle(f'Test Data $width$ Prediction vs Ground Truth, ' + open_loop_string, fontsize=sup_title_size)
        if t_id < max_length-1:
            plt.pause(0.1)  # pause to visualize the animation
        else:
            plt.show()
        if t_id==0:
            input("Press Enter to continue...")
            time.sleep(3)