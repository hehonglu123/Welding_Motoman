import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import pickle
import sys
import glob
sys.path.append('../')
sys.path.append('../../mocap/')
from Models import *

# set random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(data, train_index, test_index, obs_delay_t, memory_t, sample_rate, epochs=1000, min_max_dict=None):
    
    control_input_size = 2 # cmd_v and cmd_feedrate
    observation_size = 1 # width, (height)

    # Define the input size, hidden size, and output size
    input_size = int(memory_t*sample_rate*control_input_size + (memory_t-obs_delay_t)*sample_rate*observation_size) # all past control inputs and past observations
    print("input size:", input_size)
    hidden_sizes = [1000,1000,1000] # hidden layer sizes
    output_size = 1 # current thermal data
    
    # Define the model
    model = NeuralNetwork(input_size, output_size, hidden_sizes=hidden_sizes)
    model = model.to(device)
    # Define loss (mean squared error) and optimizer (Adam)
    loss_fn = nn.MSELoss()
    learning_rate = 0.0001
    # Define the number of epochs
    num_epochs = epochs
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("prepare training data...")
    # prepare data
    train_data = []
    train_labels = []
    for train_id in train_index:
        this_layer_id = int(train_id[0])
        # append control inputs and observations
        this_control_input = data[this_layer_id][int(train_id[1]-memory_t*sample_rate):int(train_id[1]), -2:] # cmd_v, cmd_feedrate
        this_control_input = this_control_input.flatten()
        this_observation = data[this_layer_id][int(train_id[1]-memory_t*sample_rate):int(train_id[1]-obs_delay_t*sample_rate), 2:2+observation_size] # width, (height)
        this_observation = this_observation.flatten()
        this_data = np.append(this_control_input, this_observation)
        train_data.append(this_data)
        # append thermal data
        train_labels.append(data[this_layer_id][int(train_id[1]), 1]) # thermal data
    print("prepare testing data...")
    test_data = []
    test_labels = []
    for test_id in test_index:
        this_layer_id = int(test_id[0])
        # append control inputs and observations
        this_control_input = data[this_layer_id][int(test_id[1]-memory_t*sample_rate):int(test_id[1]), -2:] # cmd_v, cmd_feedrate
        this_control_input = this_control_input.flatten()
        this_observation = data[this_layer_id][int(test_id[1]-memory_t*sample_rate):int(test_id[1]-obs_delay_t*sample_rate), 2:2+observation_size] # width, (height)
        this_observation = this_observation.flatten()
        this_data = np.append(this_control_input, this_observation)
        test_data.append(this_data)
        # append thermal data
        test_labels.append(data[this_layer_id][int(test_id[1]), 1]) # thermal data
    # convert to numpy arrays and then to torch tensors
    train_data = np.array(train_data, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.float32)
    test_data = np.array(test_data, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.float32)
    print("test label shapes:", test_labels.shape)
    train_data = torch.from_numpy(train_data).to(device)
    train_labels = torch.from_numpy(train_labels).to(device)
    train_labels = train_labels.view(-1, 1) # reshape to (N, 1)
    test_data = torch.from_numpy(test_data).to(device)
    test_labels = torch.from_numpy(test_labels).to(device)
    test_labels = test_labels.view(-1, 1) # reshape to (N, 1)

    time_start = time.perf_counter()
    ### training loop ###
    training_loss_all = []
    validation_loss_all = []
    for epoch in range(num_epochs):
        # Forward pass
        train_output_pred = model(train_data)
        # Compute the loss
        loss = loss_fn(train_output_pred, train_labels)
        # if len(training_loss_all) == 0 or loss.item() < np.min(training_loss_all):
            # save the model
            # torch.save(model.state_dict(), model_dir+'best_training_model.pt')

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        val_output_pred = model(test_data)
        val_loss = loss_fn(val_output_pred, test_labels)
        # if len(validation_loss_all) == 0 or val_loss.item() < np.min(validation_loss_all):
        #     # save the model
        #     torch.save(model.state_dict(), model_dir+'best_validation_model.pt')

        # Store model if smaller validation loss
        if len(validation_loss_all)==0 or val_loss.item() < np.max(validation_loss_all):
            torch.save(model.state_dict(), 'best_validation_model.pt')

        # Store the losses for plotting
        training_loss_all.append(loss.item())
        validation_loss_all.append(val_loss.item())

        if epoch % int(epochs/10) == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')
            print(f'Before normalization loss:', loss.item()*(min_max_dict['thermal'][1]-min_max_dict['thermal'][0]))
    print(f'Training time: {time.perf_counter()-time_start:.2f} seconds')
    ### plot losses ###
    plt.plot(training_loss_all, label='Training Loss')
    plt.plot(validation_loss_all, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid()
    plt.show()

    # plot the prediction of a layer vs the ground truth
    data_labels_all = []
    data_output_all = []
    for layer_id in range(len(data)):
        test_data_layer = test_data[test_index[:,0] == layer_id]
        test_labels_layer = test_labels[test_index[:,0] == layer_id]
        test_output_pred_layer = model(test_data_layer)
        test_output_pred_layer = test_output_pred_layer.cpu().detach().numpy()
        test_labels_layer = test_labels_layer.cpu().detach().numpy()
        train_data_layer = train_data[train_index[:,0] == layer_id]
        train_labels_layer = train_labels[train_index[:,0] == layer_id]
        train_output_pred_layer = model(train_data_layer)
        train_output_pred_layer = train_output_pred_layer.cpu().detach().numpy()
        train_labels_layer = train_labels_layer.cpu().detach().numpy()

        test_output_pred_layer = test_output_pred_layer*(min_max_dict['thermal'][1]-min_max_dict['thermal'][0])+min_max_dict['thermal'][0]
        test_labels_layer = test_labels_layer*(min_max_dict['thermal'][1]-min_max_dict['thermal'][0])+min_max_dict['thermal'][0]
        train_output_pred_layer = train_output_pred_layer*(min_max_dict['thermal'][1]-min_max_dict['thermal'][0])+min_max_dict['thermal'][0]
        train_labels_layer = train_labels_layer*(min_max_dict['thermal'][1]-min_max_dict['thermal'][0])+min_max_dict['thermal'][0]
        data_labels_all.extend(test_labels_layer)
        data_output_all.extend(test_output_pred_layer)
        data_labels_all.extend(train_labels_layer)
        data_output_all.extend(train_output_pred_layer)
    data_labels_all = np.array(data_labels_all).flatten()
    data_output_all = np.array(data_output_all).flatten()

    data_labels_all_sort_id = np.argsort(data_labels_all)
    print(data_labels_all.shape)
    data_labels_all = data_labels_all[data_labels_all_sort_id]
    data_output_all = data_output_all[data_labels_all_sort_id]

    plt.plot(data_output_all, label='Prediction')
    plt.plot(data_labels_all, label='Ground Truth')
    plt.xlabel('Sample')
    plt.ylabel('Thermal Data')
    plt.title('Thermal Data Prediction')
    plt.legend()
    plt.grid()
    plt.show()

    data_error_all = data_output_all - data_labels_all
    data_error_all = np.abs(data_error_all)

    print("Mean error:", np.mean(data_error_all))
    
    data_lam_hat = 1/np.mean(data_error_all)
    data_exp_dist = stats.expon(scale=1/data_lam_hat)
    plt.hist(data_error_all, bins=100)
    plt.plot(data_exp_dist.pdf(np.linspace(0, 0.1, 100)), label='Exponential Distribution', color='red')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.legend()
    plt.grid()
    plt.show()

    

if __name__ == "__main__":
    # load data
    data_dir = '../../data/wall_weld_test/'
    logdata_dir_all = ['weld_fujiscan_2025_02_26_18_08_18/','weld_fujiscan_2025_02_26_16_24_21/']

    # parameters
    obs_delay_t = 2 # sec, delay because of the fuji cam delay scanning
    sample_rate = 30 # Hz, using the rate of ir camera
    memory_t = 4 # sec, how long the model can remember, larger than obs_delay_t
    train_test_split = 0.8 # 80% for training, 20% for testing
    epochs = 10000 # number of epochs for training

    thermal_min = 9500 # min thermal data, subject to changes
    thermal_max = 23000 # max thermal data, subject to changes
    feedrate_min = 100 # min feedrate
    feedrate_max = 240 # max feedrate, subject to changes
    v_min = 5 # min cmd_v, subject to changes
    v_max = 10 # max cmd_v, subject to changes
    min_max_dict = {'thermal':[thermal_min, thermal_max], 'feedrate':[feedrate_min, feedrate_max], 'v':[v_min, v_max]}

    ignore_start_end = 10
    start_x = -55 + ignore_start_end
    end_x = 55 - ignore_start_end

    # prepare data
    data = []
    train_data_index = []
    test_data_index = []
    for logdata_dir_name in logdata_dir_all:
        logdata_dir = data_dir + logdata_dir_name

        with open(logdata_dir+'weld_meta_data.yml', 'r') as f:
            meta_data = yaml.safe_load(f)
        
        total_layers_name = glob.glob(logdata_dir+'layer*')
        layer_nums = []
        for layer_name in total_layers_name:
            this_layer = layer_name.split('\\')[-1]
            this_layer = this_layer.split('r')[-1]
            layer_nums.append(int(this_layer))
        layer_nums = np.sort(layer_nums)
        
        for lauer_n_id,layer_n in enumerate(layer_nums):
            this_layer_dir = logdata_dir+'layer'+str(layer_n)+'/'

            try:
                weld_data_sample = np.loadtxt(this_layer_dir + 'profile_welding_'+str(sample_rate)+'.csv', delimiter=',', skiprows=1)
            except FileNotFoundError:
                weld_data = pd.read_csv(this_layer_dir + 'profile_welding.csv', header=0)
                weld_data = weld_data.to_dict(orient='list')

                x_location = np.array(weld_data['x'])

                if x_location[-1]>x_location[0]:
                    start_index = np.where(x_location > start_x)[0][0] 
                    end_index = np.where(x_location < end_x)[0][-1]
                else:
                    start_index = np.where(x_location < end_x)[0][0] 
                    end_index = np.where(x_location > start_x)[0][-1]
                
                timestamps = np.array(weld_data['# time'])[start_index:end_index]
                width_data = np.array(weld_data['width'])[start_index:end_index]
                height_data = np.array(weld_data['height'])[start_index:end_index]
                thermal_data = np.array(weld_data['thermal'])[start_index:end_index]
                cmd_v = np.array(weld_data['cmd_v'])[start_index:end_index]
                cmd_feedrate = np.array(weld_data['cmd_feedrate'])[start_index:end_index]

                # remove index where width_data is 0
                thermal_data = np.delete(thermal_data, np.where(width_data < 0.01)[0])
                height_data = np.delete(height_data, np.where(width_data < 0.01)[0])
                cmd_v = np.delete(cmd_v, np.where(width_data < 0.01)[0])
                cmd_feedrate = np.delete(cmd_feedrate, np.where(width_data < 0.01)[0])
                timestamps = np.delete(timestamps, np.where(width_data < 0.01)[0])
                width_data = np.delete(width_data, np.where(width_data < 0.01)[0])
                print("len timestamps:", len(timestamps))

                timestamps_interp = np.arange(timestamps[0], timestamps[-1], 1/sample_rate)
                # thermal_data_interp = np.interp(timestamps_interp, timestamps, thermal_data)
                # width_data_interp = np.interp(timestamps_interp, timestamps, width_data)
                # cmd_v_interp = np.interp(timestamps_interp, timestamps, cmd_v)
                # cmd_feedrate_interp = np.interp(timestamps_interp, timestamps, cmd_feedrate)
                # average width over the time window
                thermal_data_interp = np.zeros_like(timestamps_interp)
                width_data_interp = np.zeros_like(timestamps_interp)
                height_data_interp = np.zeros_like(timestamps_interp)
                cmd_v_interp = np.zeros_like(timestamps_interp)
                cmd_feedrate_interp = np.zeros_like(timestamps_interp)
                for i in range(len(timestamps_interp)):
                    # window_id_start = np.where(timestamps >= timestamps_interp[i]-sample_rate/4)[0][0]
                    # window_id_end = np.where(timestamps <= timestamps_interp[i]+sample_rate/4)[0][-1]
                    window_id_start = np.where(timestamps >= timestamps_interp[i]-1/sample_rate)[0][0]
                    window_id_end = np.where(timestamps <= timestamps_interp[i])[0][-1]
                    if window_id_end<=window_id_start:
                        print("window_id_start:", window_id_start)
                        print(window_id_end)
                        window_id_end = window_id_start + 1
                        # plt.plot(np.diff(timestamps), label='diff time')
                        # plt.show()
                    if window_id_end>len(timestamps) or window_id_start>=len(timestamps):
                        print("Window id out of range")
                        print("window_id_start:", window_id_start)
                        print(window_id_end)
                        window_id_start = len(timestamps)-1
                        window_id_end = len(timestamps)
                    thermal_data_interp[i] = np.mean(thermal_data[window_id_start:window_id_end])
                    if np.isnan(thermal_data_interp[i]):
                        print("thermal data is nan")
                        print("window_id_start:", window_id_start)
                        print(window_id_end)
                        print("thermal data len:", len(thermal_data))
                    width_data_interp[i] = np.mean(width_data[window_id_start:window_id_end])
                    height_data_interp[i] = np.mean(height_data[window_id_start:window_id_end])
                    cmd_v_interp[i] = np.mean(cmd_v[window_id_start:window_id_end])
                    cmd_feedrate_interp[i] = np.mean(cmd_feedrate[window_id_start:window_id_end])
                # remove nan values with the mean of the neighbors
                width_data_interp = np.where(np.isnan(width_data_interp), np.nanmean(width_data_interp), width_data_interp)
                
                data_interp = {}
                data_interp['timestamps'] = timestamps_interp
                data_interp['thermal'] = thermal_data_interp
                data_interp['width'] = width_data_interp
                data_interp['height'] = height_data_interp
                data_interp['cmd_v'] = cmd_v_interp
                data_interp['cmd_feedrate'] = cmd_feedrate_interp
                header = ['timestamps', 'thermal', 'width', 'height', 'cmd_v', 'cmd_feedrate']
                data_interp = pd.DataFrame(data_interp, columns=header)
                data_interp.to_csv(this_layer_dir + 'profile_welding_'+str(sample_rate)+'.csv', index=False, header=True)
                print(this_layer_dir + 'profile_welding_'+str(sample_rate)+'.csv')
                weld_data_sample = np.loadtxt(this_layer_dir + 'profile_welding_'+str(sample_rate)+'.csv', delimiter=',', skiprows=1)

            # normalize data
            weld_data_sample[:,1] = (weld_data_sample[:,1] - thermal_min) / (thermal_max - thermal_min)
            # print("thermal data min:", np.min(weld_data_sample[:,1]), "max:", np.max(weld_data_sample[:,1]))
            weld_data_sample[:,4] = (weld_data_sample[:,4] - v_min) / (v_max - v_min)
            # print("cmd_v min:", np.min(weld_data_sample[:,4]), "max:", np.max(weld_data_sample[:,4]))
            weld_data_sample[:,5] = (weld_data_sample[:,5] - feedrate_min) / (feedrate_max - feedrate_min)
            # print("cmd_feedrate min:", np.min(weld_data_sample[:,5]), "max:", np.max(weld_data_sample[:,5]))
            # weld_data_sample = np.clip(weld_data_sample, 0, 1) # clip data to [0, 1]

            # draw height and thermal on the same plot with different y axis
            # fig, ax1 = plt.subplots()
            # ax1.plot(thermal_data_plot, 'r-', label='thermal data')
            # ax1.set_ylabel('thermal data', color='r')
            # ax1.tick_params(axis='y', labelcolor='r')
            # ax2 = ax1.twinx()
            # ax2.plot(height_data_plot, 'b-', label='height data')
            # ax2.set_ylabel('height data', color='b')
            # ax2.tick_params(axis='y', labelcolor='b')
            # fig.tight_layout()
            # plt.title('thermal data and height data')
            # plt.show()

            
            # load data and split
            data.append(weld_data_sample)
            data_available_index = int(weld_data_sample.shape[0] - memory_t*sample_rate)
            this_train_data_index = np.random.choice(data_available_index, int(data_available_index*train_test_split), replace=False) + memory_t*sample_rate
            this_test_data_index = np.setdiff1d(np.arange(data_available_index), this_train_data_index) + memory_t*sample_rate
            train_data_index.extend(np.vstack((np.repeat(len(data)-1, len(this_train_data_index)),this_train_data_index)).T)
            test_data_index.extend(np.vstack((np.repeat(len(data)-1, len(this_test_data_index)),this_test_data_index)).T)

    
    # thermal_data_plot = []
    # width_data_plot = []
    # height_data_plot = []
    # for layer_id in range(len(data)):
    #     thermal_data_plot.extend(deepcopy(data[layer_id][:,1]))
    #     width_data_plot.extend(deepcopy(data[layer_id][:,2]))
    #     height_data_plot.extend(deepcopy(data[layer_id][:,3]))
    # thermal_data_plot = np.array(thermal_data_plot)
    # width_data_plot = np.array(width_data_plot)
    # height_data_plot = np.array(height_data_plot)
    # thermal_data_plot_sort_id = np.argsort(thermal_data_plot)
    # thermal_data_plot = thermal_data_plot[thermal_data_plot_sort_id]
    # width_data_plot = width_data_plot[thermal_data_plot_sort_id]
    # height_data_plot = height_data_plot[thermal_data_plot_sort_id]
    # # draw width and thermal on the same plot with different y axis
    # fig, ax1 = plt.subplots()
    # ax1.plot(thermal_data_plot, 'r-', label='thermal data')
    # ax1.set_ylabel('thermal data', color='r')
    # ax1.tick_params(axis='y', labelcolor='r')
    # ax2 = ax1.twinx()
    # ax2.plot(width_data_plot, 'b-', label='width data')
    # ax2.set_ylabel('width data', color='b')
    # ax2.tick_params(axis='y', labelcolor='b')
    # fig.tight_layout()
    # plt.title('thermal data and width data')
    # plt.show()
    
    train_data_index = np.array(train_data_index)
    test_data_index = np.array(test_data_index)
    print("train data index:", train_data_index)

    # shuffle data
    # np.random.shuffle(train_data_index)
    # np.random.shuffle(test_data_index)

    # train model
    train_model(data=data, train_index=train_data_index, test_index=test_data_index, obs_delay_t=obs_delay_t, memory_t=memory_t, sample_rate=sample_rate, epochs=epochs, min_max_dict=min_max_dict)
