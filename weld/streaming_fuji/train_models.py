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
sys.path.append('../')
sys.path.append('../../mocap/')
from Models import *

inch2mm = 25.4
mm2inch = 1/inch2mm

# for plotting
xy_label_size = 14
xy_tick_size = 12
legend_size = 12
title_size = 16
sup_title_size = 18

# static randomize seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

def plot_vt_vw_dh_dw(v_torch_range, v_wire_range, dh_pred, dw_pred, plot_title='Predicted dh and dw'):

    # make the v_torch_range and v_wire_range 2D arrays for imshow
    v_torch_range_grid = np.repeat(v_torch_range, len(v_wire_range)).reshape(len(v_torch_range), len(v_wire_range))
    v_wire_range_grid = np.tile(v_wire_range, len(v_torch_range)).reshape(len(v_torch_range), len(v_wire_range))
    v_wire_range_ipm = v_wire_range * mm2inch * 60 # convert mm/s to ipm

    
    # plot dh and dw using colormap and imshow
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # dh
    sc = ax[0].scatter(v_torch_range_grid, v_wire_range_grid, c=dh_pred, cmap='viridis', marker='o')
    ax[0].set_xlabel('Torch Speed (mm/s)', fontsize=xy_label_size)
    ax[0].tick_params(axis='x', labelsize=xy_tick_size)
    ax[0].set_ylabel('Wire Feedrate (ipm)', fontsize=xy_label_size)
    ax[0].set_yticks(np.arange(min(v_wire_range_ipm), max(v_wire_range_ipm)+1, 20) * inch2mm / 60)
    ax[0].set_yticklabels(np.arange(min(v_wire_range_ipm), max(v_wire_range_ipm)+1, 20).astype(int), fontsize=xy_tick_size)
    ax[0].set_title('Predicted $\Delta h$', fontsize=title_size)
    ax[0].set_aspect('auto')
    ax[0].grid()
    cbar = plt.colorbar(sc, ax=ax[0])
    cbar.set_label('$\Delta h$ (mm)', fontsize=xy_label_size)
    cbar.ax.tick_params(labelsize=xy_tick_size)
    # dw
    sc = ax[1].scatter(v_torch_range_grid, v_wire_range_grid, c=dw_pred, cmap='viridis', marker='o')
    ax[1].set_xlabel('Torch Speed (mm/s)', fontsize=xy_label_size)
    ax[1].tick_params(axis='x', labelsize=xy_tick_size)
    ax[1].set_ylabel('Wire Feedrate (ipm)', fontsize=xy_label_size)
    ax[1].set_yticks(np.arange(min(v_wire_range_ipm), max(v_wire_range_ipm)+1, 20) * inch2mm / 60)
    ax[1].set_yticklabels(np.arange(min(v_wire_range_ipm), max(v_wire_range_ipm)+1, 20).astype(int), fontsize=xy_tick_size)
    ax[1].set_title('Predicted $\Delta w$', fontsize=title_size)
    ax[1].set_aspect('auto')
    ax[1].grid()
    cbar = plt.colorbar(sc, ax=ax[1])
    cbar.set_label('$\Delta w$ (mm)', fontsize=xy_label_size)
    cbar.ax.tick_params(labelsize=xy_tick_size)
    plt.suptitle(plot_title, fontsize=sup_title_size)
    plt.tight_layout()
    plt.show()

def plot_error_distribution_all(dh_error_all, dw_error_all, plot_title='Error Distribution'):

    # plot the error distribution
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    cmap = plt.get_cmap('tab10')

    for i,key in enumerate(dh_error_all.keys()):
        dh_lambda_hat = 1 / np.mean(dh_error_all[key])
        dh_exp_dist = stats.expon(scale=1/dh_lambda_hat)
        dw_lambda_hat = 1 / np.mean(dw_error_all[key])
        dw_exp_dist = stats.expon(scale=1/dw_lambda_hat)

        ax[0].hist(dh_error_all[key], bins=50, density=True, alpha=0.5, label=key, color=cmap(i))
        ax[0].plot(np.linspace(0, np.max(dh_error_all[key]), 100), dh_exp_dist.pdf(np.linspace(0, np.max(dh_error_all[key]), 100)), label=key+' pdf', color=cmap(i))
        ax[0].set_xlabel('$\Delta h$ Error (mm)', fontsize=xy_label_size)
        ax[0].set_ylabel('Density', fontsize=xy_label_size)
        ax[0].set_title('$\Delta h$ Error Distribution', fontsize=title_size)
        ax[0].legend(fontsize=legend_size)
        ax[0].tick_params(axis='x', labelsize=xy_tick_size)
        ax[0].tick_params(axis='y', labelsize=xy_tick_size)
        ax[0].grid()

        ax[1].hist(dw_error_all[key], bins=50, density=True, alpha=0.5, label=key, color=cmap(i))
        ax[1].plot(np.linspace(0, np.max(dw_error_all[key]), 100), dw_exp_dist.pdf(np.linspace(0, np.max(dw_error_all[key]), 100)), label=key+' pdf', color=cmap(i))
        ax[1].set_xlabel('$\Delta w$ Error (mm)', fontsize=xy_label_size)
        ax[1].set_ylabel('Density', fontsize=xy_label_size)
        ax[1].set_title('$\Delta w$ Error Distribution', fontsize=title_size)
        ax[1].legend(fontsize=legend_size)
        ax[1].tick_params(axis='x', labelsize=xy_tick_size)
        ax[1].tick_params(axis='y', labelsize=xy_tick_size)
        ax[1].grid()
    
    ax[0].set_xlim(-0.2, 1.2)
    ax[1].set_xlim(-0.2, 2.2)
    plt.suptitle(plot_title, fontsize=sup_title_size)
    plt.tight_layout()
    plt.show()

def plot_error_distribution(dh_train_error, dw_train_error, dh_val_error, dw_val_error, plot_title='Error Distribution'):

    # fit distribution with exponential distribution
    dh_lambda_hat_train = 1 / np.mean(dh_train_error)
    dh_exp_dist_train = stats.expon(scale=1/dh_lambda_hat_train)
    dh_lambda_hat_val = 1 / np.mean(dh_val_error)
    dh_exp_dist_val = stats.expon(scale=1/dh_lambda_hat_val)
    dw_lambda_hat_train = 1 / np.mean(dw_train_error)
    dw_exp_dist_train = stats.expon(scale=1/dw_lambda_hat_train)
    dw_lambda_hat_val = 1 / np.mean(dw_val_error)
    dw_exp_dist_val = stats.expon(scale=1/dw_lambda_hat_val)
    # fit distribution with half-normal distribution
    # dh_loc_hat, dh_sigma_hat = stats.halfnorm.fit(dh_train_error, floc=0)
    # dh_halfnorm_dist = stats.halfnorm(loc=dh_loc_hat, scale=dh_sigma_hat)
    # dw_loc_hat, dw_sigma_hat = stats.halfnorm.fit(dw_train_error, floc=0)
    # dw_halfnorm_dist = stats.halfnorm(loc=dw_loc_hat, scale=dw_sigma_hat)

    # plot the error distribution
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # dh train error
    ax[0].hist(dh_train_error, bins=50, density=True, alpha=0.5, label='Train')
    # dh validation error
    ax[0].hist(dh_val_error, bins=50, density=True, alpha=0.5, label='Test')
    ax[0].plot(np.linspace(0, np.max(dh_train_error), 100), dh_exp_dist_train.pdf(np.linspace(0, np.max(dh_train_error), 100)), label='Train pdf')
    # ax[0].plot(np.linspace(0, np.max(dh_train_error), 100), dh_halfnorm_dist.pdf(np.linspace(0, np.max(dh_train_error), 100)), label='Half-Normal Fit')
    ax[0].plot(np.linspace(0, np.max(dh_train_error), 100), dh_exp_dist_val.pdf(np.linspace(0, np.max(dh_train_error), 100)), label='Test pdf')
    ax[0].set_xlabel('$\Delta h$ Error (mm)', fontsize=xy_label_size)
    ax[0].set_ylabel('Density', fontsize=xy_label_size)
    ax[0].set_title('$\Delta h$ Error Distribution', fontsize=title_size)
    ax[0].legend(fontsize=legend_size)
    ax[0].tick_params(axis='x', labelsize=xy_tick_size)
    ax[0].tick_params(axis='y', labelsize=xy_tick_size)
    ax[0].grid()
    # dw train error 
    ax[1].hist(dw_train_error, bins=50, density=True, alpha=0.5, label='Train')
    # dw validation error
    ax[1].hist(dw_val_error, bins=50, density=True, alpha=0.5, label='Test')
    ax[1].plot(np.linspace(0, np.max(dw_train_error), 100), dw_exp_dist_train.pdf(np.linspace(0, np.max(dw_train_error), 100)), label='Train pdf')
    # ax[1].plot(np.linspace(0, np.max(dw_train_error), 100), dw_halfnorm_dist.pdf(np.linspace(0, np.max(dw_train_error), 100)), label='Half-Normal Fit')
    ax[1].plot(np.linspace(0, np.max(dw_train_error), 100), dw_exp_dist_val.pdf(np.linspace(0, np.max(dw_train_error), 100)), label='Test pdf')
    ax[1].set_xlabel('$\Delta w$ Error (mm)', fontsize=xy_label_size)
    ax[1].set_ylabel('Density', fontsize=xy_label_size)
    ax[1].set_title('$\Delta w$ Error Distribution', fontsize=title_size)
    ax[1].legend(fontsize=legend_size)
    ax[1].grid()
    ax[1].tick_params(axis='x', labelsize=xy_tick_size)
    ax[1].tick_params(axis='y', labelsize=xy_tick_size)
    plt.suptitle(plot_title, fontsize=sup_title_size)
    plt.tight_layout()
    plt.show()

    return dh_exp_dist_train.interval(0.95), dw_exp_dist_train.interval(0.95)

def plot_error_heatmap(dh_train_error, dw_train_error, dh_val_error, dw_val_error, train_inputs, val_inputs, plot_title='Error Heatmap'):

    dh_error_all = np.abs(np.concatenate((dh_train_error, dh_val_error)))
    dw_error_all = np.abs(np.concatenate((dw_train_error, dw_val_error)))
    inputs_all = np.concatenate((train_inputs, val_inputs))
    dw_inputs_all_ipm = inputs_all[:, 1] * mm2inch * 60 # convert mm/s to ipm

    ## 3D bar error heatmap
    # fig = plt.figure(1, 2)
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.bar3d(inputs_all[:, 0], dw_inputs_all_ipm, np.zeros_like(dh_error_all), 0.2, 0.2, dh_error_all, shade=True)
    # # use error as colormap with bar3d
    # ax.bar3d(inputs_all[:, 0], inputs_all[:, 1], np.zeros_like(dh_error_all), 0.2, 0.2, dh_error_all, shade=True, color=plt.cm.viridis(dh_error_all/np.max[0](dh_error_all)))
    # ax.set_xlabel('Torch Speed (mm/s)', fontsize=xy_label_size)
    # ax.set_ylabel('Wire Feedrate (ipm)', fontsize=xy_label_size)
    # ax.set_zlabel('$\Delta h$ Error (mm)', fontsize=xy_label_size)
    # ax.set_title('$\Delta h$ Error Heatmap', fontsize=title_size)
    # ax.set_xticks(np.arange(min(inputs_all[:, 0]), max(inputs_all[:, 0])+1, 2))
    # ax.set_yticks(np.arange(min(dw_inputs_all_ipm), max(dw_inputs_all_ipm)+1, 20) * inch2mm / 60)
    # ax.set_yticklabels(np.arange(min(dw_inputs_all_ipm), max(dw_inputs_all_ipm)+1, 20).astype(int), fontsize=xy_tick_size)
    # ax.set_zticks(np.arange(0, np.max(dh_error_all)+1, 0.5))
    # ax.set_zticklabels(np.arange(0, np.max(dh_error_all)+1, 0.5).astype(int), fontsize=xy_tick_size)
    # ax.set_box_aspect([1, 1, 0.5])  # aspect ratio is 1:1:0.5
    # plt.title(plot_title, fontsize=sup_title_size)
    # plt.tight_layout()
    # plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].scatter(inputs_all[:, 0], inputs_all[:, 1], c=dh_error_all, cmap='viridis', marker='o')
    ax[0].set_xlabel('Torch Speed (mm/s)', fontsize=xy_label_size)
    ax[0].set_ylabel('Wire Feedrate (ipm)', fontsize=xy_label_size)
    ax[0].set_yticks(np.arange(min(dw_inputs_all_ipm), max(dw_inputs_all_ipm)+1, 20) * inch2mm / 60)
    ax[0].set_yticklabels(np.arange(min(dw_inputs_all_ipm), max(dw_inputs_all_ipm)+1, 20).astype(int), fontsize=xy_tick_size)
    ax[0].set_title('$\Delta h$ Error Heatmap', fontsize=title_size)
    ax[0].set_aspect('auto')
    ax[0].grid()
    cbar = plt.colorbar(ax[0].collections[0], ax=ax[0])
    cbar.set_label('$\Delta h$ Error (mm)', fontsize=xy_label_size)
    cbar.ax.tick_params(labelsize=xy_tick_size)
    # dw
    ax[1].scatter(inputs_all[:, 0], inputs_all[:, 1], c=dw_error_all, cmap='viridis', marker='o')
    ax[1].set_xlabel('Torch Speed (mm/s)', fontsize=xy_label_size)
    ax[1].set_ylabel('Wire Feedrate (ipm)', fontsize=xy_label_size)
    ax[1].set_yticks(np.arange(min(dw_inputs_all_ipm), max(dw_inputs_all_ipm)+1, 20) * inch2mm / 60)
    ax[1].set_yticklabels(np.arange(min(dw_inputs_all_ipm), max(dw_inputs_all_ipm)+1, 20).astype(int), fontsize=xy_tick_size)
    ax[1].set_title('$\Delta w$ Error Heatmap', fontsize=title_size)
    ax[1].set_aspect('auto')
    ax[1].grid()
    cbar = plt.colorbar(ax[1].collections[0], ax=ax[1])
    cbar.set_label('$\Delta w$ Error (mm)', fontsize=xy_label_size)
    cbar.ax.tick_params(labelsize=xy_tick_size)
    plt.suptitle(plot_title, fontsize=sup_title_size)
    plt.tight_layout()
    plt.show()

def get_rmse(error_array):

    error_array = np.array(error_array).flatten()

    assert error_array.ndim == 1, "Error array must be 1D"
    assert error_array.size > 0, "Error array must not be empty"

    return np.sqrt(np.mean(error_array**2))

def train_loglog(train_input,train_output,val_input,val_output,quadratic=False):

    train_input = np.array(train_input)[:,:2]
    val_input = np.array(val_input)[:,:2]
    train_input_log = np.log(train_input)
    train_output_log = np.log(train_output)
    val_input_log = np.log(val_input)
    val_output_log = np.log(val_output)

    if quadratic:
        mat_A_train = np.hstack((train_input_log[:,0].reshape(-1,1)*train_input_log[:,1].reshape(-1,1), train_input_log, np.ones_like(train_input_log[:,0].reshape(-1,1))))
        mat_A_val = np.hstack((val_input_log[:,0].reshape(-1,1)*val_input_log[:,1].reshape(-1,1), val_input_log, np.ones_like(val_input_log[:,0].reshape(-1,1))))
    else:
        mat_A_train = np.hstack((train_input_log, np.ones_like(train_input_log[:,0].reshape(-1,1))))
        mat_A_val = np.hstack((val_input_log, np.ones_like(val_input_log[:,0].reshape(-1,1))))
    mat_B_dh = train_output_log[:,0].reshape(-1,1)
    mat_B_dw = train_output_log[:,1].reshape(-1,1)

    # train the linear model
    time_start = time.perf_counter()
    theta_param_dh = np.linalg.pinv(mat_A_train)@mat_B_dh
    theta_param_dw = np.linalg.pinv(mat_A_train)@mat_B_dw
    print(f'Training time: {time.perf_counter()-time_start:.2f} seconds')

    # prediction
    dh_train_pred = (mat_A_train@theta_param_dh).flatten()
    dw_train_pred = (mat_A_train@theta_param_dw).flatten()
    dh_val_pred = (mat_A_val@theta_param_dh).flatten()
    dw_val_pred = (mat_A_val@theta_param_dw).flatten()

    # predict the output using the linear model and statistic data
    train_dh_error = np.exp(train_output_log[:,0])-np.exp(dh_train_pred)
    train_dw_error = np.exp(train_output_log[:,1])-np.exp(dw_train_pred)
    val_dh_error = np.exp(val_output_log[:,0])-np.exp(dh_val_pred)
    val_dw_error = np.exp(val_output_log[:,1])-np.exp(dw_val_pred)

    # visualize the prediction
    v_torch_range = np.arange(2, 13, 0.2)
    v_wire_range = np.arange(100, 201, 1) * inch2mm / 60 # ipm to mm/s
    dh_pred = np.zeros((len(v_torch_range), len(v_wire_range)))
    dw_pred = np.zeros((len(v_torch_range), len(v_wire_range)))
    for i, v_torch in enumerate(v_torch_range):
        for j, v_wire in enumerate(v_wire_range):
            input_data = np.log(np.array([v_torch, v_wire]))
            if quadratic:
                mat_A = np.hstack((input_data[0]*input_data[1], input_data, np.ones_like(input_data[0])))
            else:
                mat_A = np.hstack((input_data, np.ones_like(input_data[0])))
            dh_pred[i,j] = np.exp(mat_A@theta_param_dh)[0]
            dw_pred[i,j] = np.exp(mat_A@theta_param_dw)[0]
    dh_pred = dh_pred.flatten()
    dw_pred = dw_pred.flatten()
    # plot the predicted dh and dw using colormap and imshow
    plot_title = 'Predicted $\Delta h$ and $\Delta w$ (Linear Model)' if not quadratic else 'Predicted $\Delta h$ and $\Delta w$ (Quadratic Model)'
    plot_vt_vw_dh_dw(v_torch_range, v_wire_range, dh_pred, dw_pred, plot_title=plot_title)
    
    return train_dh_error, train_dw_error, val_dh_error, val_dw_error
    
def train_NN(train_input,train_output,val_input,val_output,torch_height=True,layer_height=False,train_model=True,model_dir=''):

    input_chosen = [0,1]
    input_chosen.append(2) if torch_height else None
    input_chosen.append(3) if layer_height else None
    input_chosen = np.array(input_chosen)

    # convert to torch tensors
    train_input = torch.tensor(train_input[:,input_chosen], dtype=torch.float32)
    train_output = torch.tensor(train_output, dtype=torch.float32)
    val_input = torch.tensor(val_input[:,input_chosen], dtype=torch.float32)
    val_output = torch.tensor(val_output, dtype=torch.float32)

    # Define the input size, hidden size, and output size
    input_size = train_input.shape[1]
    hidden_sizes = [100,100,100]
    output_size = train_output.shape[1]
    # Define the model
    model = NeuralNetwork(input_size, output_size, hidden_sizes=hidden_sizes)
    # Define loss (mean squared error) and optimizer (Adam)
    loss_fn = nn.MSELoss()
    learning_rate = 0.0001
    # Define the number of epochs
    num_epochs = 10000
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if train_model:
        time_start = time.perf_counter()
        ### training loop ###
        training_loss_all = []
        validation_loss_all = []
        for epoch in range(num_epochs):
            # Forward pass
            train_output_pred = model(train_input)
            # Compute the loss
            loss = loss_fn(train_output_pred, train_output)
            if len(training_loss_all) == 0 or loss.item() < np.min(training_loss_all):
                # save the model
                torch.save(model.state_dict(), model_dir+'best_training_model.pt')

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation
            val_output_pred = model(val_input)
            val_loss = loss_fn(val_output_pred, val_output)
            if len(validation_loss_all) == 0 or val_loss.item() < np.min(validation_loss_all):
                # save the model
                torch.save(model.state_dict(), model_dir+'best_validation_model.pt')

            # Store the losses for plotting
            training_loss_all.append(loss.item())
            validation_loss_all.append(val_loss.item())

            if epoch % 1000 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')
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
    # load the model (use best validation model for evaluation)
    model.load_state_dict(torch.load(model_dir+'best_validation_model.pt',weights_only=True))
    print("Model loaded from: ", model_dir+'best_validation_model.pt')
    # final training loss after last epoch
    train_output_pred = model(train_input)
    val_output_pred = model(val_input)
    loss = loss_fn(train_output_pred, train_output)
    val_loss = loss_fn(val_output_pred, val_output)
    print("Final Training Loss: ", loss.item())
    print("Final Validation Loss: ", val_loss.item())
    dh_error_train = train_output_pred[:,0].detach().numpy() - train_output[:,0].detach().numpy()
    dw_error_train = train_output_pred[:,1].detach().numpy() - train_output[:,1].detach().numpy()
    dh_error_val = val_output_pred[:,0].detach().numpy() - val_output[:,0].detach().numpy()
    dw_error_val = val_output_pred[:,1].detach().numpy() - val_output[:,1].detach().numpy()

    ### plot predictions ###
    ### v_torch = 2~12 mm/sec with 1mm/sec step, 12 included
    ### v_wire = 100~200 ipm with 10ipm step, 200 included
    v_torch_range = np.arange(2, 13, 0.2)
    v_wire_range = np.arange(100, 201, 1) * inch2mm / 60 # ipm to mm/s
    if torch_height and layer_height:
        torch_height_ave = np.mean(train_input.detach().numpy()[:,2])
        layer_height_ave = np.mean(train_input.detach().numpy()[:,3])
        torch_layer_height_ave = np.array([torch_height_ave, layer_height_ave])
    elif torch_height or layer_height:
        torch_layer_height_ave = np.array([np.mean(train_input.detach().numpy()[:,2])])
    else:
        torch_layer_height_ave = np.array([])
    dh_pred = np.zeros((len(v_torch_range), len(v_wire_range)))
    dw_pred = np.zeros((len(v_torch_range), len(v_wire_range)))
    for i, v_torch in enumerate(v_torch_range):
        for j, v_wire in enumerate(v_wire_range):
            input_data = torch.tensor(np.append(np.array([v_torch, v_wire]),torch_layer_height_ave), dtype=torch.float32).unsqueeze(0) # add batch dimension
            with torch.no_grad():
                output_data = model(input_data)
            dh_pred[i,j], dw_pred[i,j] = output_data.squeeze().numpy()
    dh_pred = dh_pred.flatten()
    dw_pred = dw_pred.flatten()
    # plot the predicted dh and dw using colormap and imshow
    plot_vt_vw_dh_dw(v_torch_range, v_wire_range, dh_pred, dw_pred, plot_title='Predicted $\Delta h$ and $\Delta w$ (NN Model)')

    return dh_error_train, dw_error_train, dh_error_val, dw_error_val

def train_GP(train_input,train_output,val_input,val_output,torch_height=True,layer_height=False,train_model=True,model_dir=''):

    input_chosen = [0,1]
    input_chosen.append(2) if torch_height else None
    input_chosen.append(3) if layer_height else None
    input_chosen = np.array(input_chosen)

    train_input = np.array(train_input)[:,input_chosen]
    val_input = np.array(val_input)[:,input_chosen]

    input_dimension = train_input.shape[1]
    output_dimension = train_output.shape[1]

    # Define kernel: RBF (squared exponential) + White noise
    kernel = RBF(length_scale=np.ones(input_dimension)*0.2, length_scale_bounds=(1e-2, 10)) \
            + WhiteKernel(noise_level=0.3, noise_level_bounds=(1e-5, 3))

    # Instantiate and fit separate GP models for each output dimension
    time_start = time.perf_counter()
    gps = []
    if train_model:
        for m in range(output_dimension):
            gp = GaussianProcessRegressor(kernel=kernel,
                                        alpha=1e-6,
                                        n_restarts_optimizer=10,
                                        normalize_y=True)
            gp.fit(train_input, train_output[:, m])
            print(f"Optimized kernel for output {m}:", gp.kernel_)
            # Save the model
            with open(model_dir+f'gp_model_{m}.pkl', 'wb') as f:
                pickle.dump(gp, f)
            gps.append(gp)
    else:
        for m in range(output_dimension):
            with open(model_dir+f'gp_model_{m}.pkl', 'rb') as f:
                gp = pickle.load(f)
            print(f"Optimized kernel for output {m}:", gp.kernel_)
            gps.append(gp)
        
    print(f'Training time: {time.perf_counter()-time_start:.2f} seconds')

    # get dh dw training/validation error
    dh_pred_train, dh_std_train = gps[0].predict(train_input, return_std=True)
    dh_error_train = train_output[:, 0] - dh_pred_train
    dw_pred_train, dw_std_train = gps[1].predict(train_input, return_std=True)
    dw_error_train = train_output[:, 1] - dw_pred_train
    dh_pred_val, dh_std_val = gps[0].predict(val_input, return_std=True)
    dh_error_val = val_output[:, 0] - dh_pred_val
    dw_pred_val, dw_std_val = gps[1].predict(val_input, return_std=True)
    dw_error_val = val_output[:, 1] - dw_pred_val

    ### plot predictions ###
    ### v_torch = 2~12 mm/sec with 1mm/sec step, 12 included
    ### v_wire = 100~200 ipm with 10ipm step, 200 included
    v_torch_range = np.arange(2, 13, 0.2)
    v_wire_range = np.arange(100, 201, 1) * inch2mm / 60 # ipm to mm/s
    if torch_height and layer_height:
        torch_height_ave = np.mean(train_input[:,2])
        layer_height_ave = np.mean(train_input[:,3])
        torch_layer_height_ave = np.array([torch_height_ave, layer_height_ave])
    elif torch_height or layer_height:
        torch_layer_height_ave = np.array([np.mean(train_input[:,2])])
    else:
        torch_layer_height_ave = np.array([])
    dh_pred = np.zeros((len(v_torch_range), len(v_wire_range)))
    dw_pred = np.zeros((len(v_torch_range), len(v_wire_range)))
    for i, v_torch in enumerate(v_torch_range):
        for j, v_wire in enumerate(v_wire_range):
            input_data = np.append(np.array([v_torch, v_wire]),torch_layer_height_ave) # add batch dimension
            input_data = np.reshape(input_data, (1, -1))
            dh_pred[i,j] = gps[0].predict(input_data)[0]
            dw_pred[i,j] = gps[1].predict(input_data)[0]
    dh_pred = dh_pred.flatten()
    dw_pred = dw_pred.flatten()
    # plot the predicted dh and dw using colormap and imshow
    plot_vt_vw_dh_dw(v_torch_range, v_wire_range, dh_pred, dw_pred, plot_title='Predicted $\Delta h$ and $\Delta w$ (GP Model)')

    return dh_error_train, dw_error_train, dh_error_val, dw_error_val

def main():
    
    # read data from weld_data
    data_dir = '../../data/wall_weld_test/'
    weld_data = pd.read_csv(data_dir + 'weld_data/weld_data.csv', header=0)
    weld_data = weld_data.to_dict(orient='list')

    # load data to numpy arrays
    input_keys = ['v_torch','v_wire','torch_height','height']
    output_keys = ['dh','dw']
    train_input = []
    train_output = []
    for key in input_keys:
        train_input.append(np.array(weld_data[key]))
    for key in output_keys:
        train_output.append(np.array(weld_data[key]))
    train_input = np.array(train_input).T
    train_output = np.array(train_output).T

    # choose 20% of the data for validation
    training_percent = 0.8
    num_samples = train_input.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split = int(np.floor(training_percent * num_samples))
    train_indices, val_indices = indices[:split], indices[split:]
    train_input, val_input = train_input[train_indices], train_input[val_indices]
    train_output, val_output = train_output[train_indices], train_output[val_indices]

    output_string_dh = '# Error Distribution dh\n'
    output_string_dh += '| Unit (mm) | Train RMSE dh| Test RMSE dh | Train Max dh | Test Max dh | Interval 95% |\n'
    output_string_dh += '|---|---|---|---|---|---|\n'

    output_string_dw = '# Error Distribution dw\n'
    output_string_dw += '| Unit (mm) | Train RMSE dw| Test RMSE dw | Train Max dw | Test Max dw | Interval 95% |\n'
    output_string_dw += '|---|---|---|---|---|---|\n'

    # log-log model. linear
    print("Training log-log linear model...")
    dh_error_train_lnln_lin, dw_error_train_lnln_lin, dh_error_val_lnln_lin, dw_error_val_lnln_lin = \
        train_loglog(deepcopy(train_input), deepcopy(train_output), deepcopy(val_input), deepcopy(val_output),quadratic=False)
    train_rmse_dh_lnln_lin, train_rmse_dw_lnln_lin, val_rmse_dh_lnln_lin, val_rmse_dw_lnln_lin = \
        get_rmse(dh_error_train_lnln_lin), get_rmse(dw_error_train_lnln_lin), get_rmse(dh_error_val_lnln_lin), get_rmse(dw_error_val_lnln_lin)
    dh_inter_95_lnln_lin, dw_inter_95_lnln_lin = plot_error_distribution(np.abs(dh_error_train_lnln_lin), np.abs(dw_error_train_lnln_lin), np.abs(dh_error_val_lnln_lin), np.abs(dw_error_val_lnln_lin), plot_title='Error distribution (log-log linear model)') # plot training and validation error distribution for dh and dw
    plot_error_heatmap(dh_error_train_lnln_lin, dw_error_train_lnln_lin, dh_error_val_lnln_lin, dw_error_val_lnln_lin, train_input, val_input, plot_title='Error heatmap (log-log linear model)') # plot training and validation error heatmap for dh and dw
    output_string_dh += f'| Linear loglog model | {train_rmse_dh_lnln_lin:.2f} | {val_rmse_dh_lnln_lin:.2f} | {np.max(np.abs(dh_error_train_lnln_lin)):.2f} | {np.max(np.abs(dh_error_val_lnln_lin)):.2f} | {dh_inter_95_lnln_lin[0]:.2f}~{dh_inter_95_lnln_lin[1]:.2f} |\n'
    output_string_dw += f'| Linear loglog model | {train_rmse_dw_lnln_lin:.2f} | {val_rmse_dw_lnln_lin:.2f} | {np.max(np.abs(dw_error_train_lnln_lin)):.2f} | {np.max(np.abs(dw_error_val_lnln_lin)):.2f} | {dw_inter_95_lnln_lin[0]:.2f}~{dw_inter_95_lnln_lin[1]:.2f} |\n'
    # log-log model. with quadratic term
    print("Training log-log quadratic model...")
    dh_error_train_lnln_qua, dw_error_train_lnln_qua, dh_error_val_lnln_qua, dw_error_val_lnln_qua = \
        train_loglog(deepcopy(train_input), deepcopy(train_output), deepcopy(val_input), deepcopy(val_output),quadratic=True)
    train_rmse_dh_lnln_qua, train_rmse_dw_lnln_qua, val_rmse_dh_lnln_qua, val_rmse_dw_lnln_qua = \
        get_rmse(dh_error_train_lnln_qua), get_rmse(dw_error_train_lnln_qua), get_rmse(dh_error_val_lnln_qua), get_rmse(dw_error_val_lnln_qua)
    dh_inter_95_lnln_qua, dw_inter_95_lnln_qua = plot_error_distribution(np.abs(dh_error_train_lnln_qua), np.abs(dw_error_train_lnln_qua), np.abs(dh_error_val_lnln_qua), np.abs(dw_error_val_lnln_qua), plot_title='Error distribution (log-log quadratic model)') # plot training and validation error distribution for dh and dw
    plot_error_heatmap(dh_error_train_lnln_qua, dw_error_train_lnln_qua, dh_error_val_lnln_qua, dw_error_val_lnln_qua, train_input, val_input, plot_title='Error heatmap (log-log quadratic model)') # plot training and validation error heatmap for dh and dw
    output_string_dh += f'| Quadratic loglog model | {train_rmse_dh_lnln_qua:.2f} | {val_rmse_dh_lnln_qua:.2f} | {np.max(np.abs(dh_error_train_lnln_qua)):.2f} | {np.max(np.abs(dh_error_val_lnln_qua)):.2f} | {dh_inter_95_lnln_qua[0]:.2f}~{dh_inter_95_lnln_qua[1]:.2f} |\n'
    output_string_dw += f'| Quadratic loglog model | {train_rmse_dw_lnln_qua:.2f} | {val_rmse_dw_lnln_qua:.2f} | {np.max(np.abs(dw_error_train_lnln_qua)):.2f} | {np.max(np.abs(dw_error_val_lnln_qua)):.2f}  | {dw_inter_95_lnln_qua[0]:.2f}~{dw_inter_95_lnln_qua[1]:.2f} |\n'
    # Neural Network model. with torch height
    print("Training Neural Network model with torch height...")
    dh_error_train_nn, dw_error_train_nn, dh_error_val_nn, dw_error_val_nn = \
        train_NN(deepcopy(train_input), deepcopy(train_output), deepcopy(val_input), deepcopy(val_output), torch_height=True, layer_height=False, train_model=False, model_dir='weld_NN_models/')
    train_rmse_dh_nn, train_rmse_dw_nn, val_rmse_dh_nn, val_rmse_dw_nn = \
        get_rmse(dh_error_train_nn), get_rmse(dw_error_train_nn), get_rmse(dh_error_val_nn), get_rmse(dw_error_val_nn)
    dh_inter_95_nn, dw_inter_95_nn = plot_error_distribution(np.abs(dh_error_train_nn), np.abs(dw_error_train_nn), np.abs(dh_error_val_nn), np.abs(dw_error_val_nn), plot_title='Error distribution (neural network model)') # plot training and validation error distribution for dh and dw
    plot_error_heatmap(dh_error_train_nn, dw_error_train_nn, dh_error_val_nn, dw_error_val_nn, train_input, val_input, plot_title='Error heatmap (neural network model)') # plot training and validation error heatmap for dh and dw
    output_string_dh += f'| Neural Network model | {train_rmse_dh_nn:.2f} | {val_rmse_dh_nn:.2f} | {np.max(np.abs(dh_error_train_nn)):.2f} | {np.max(np.abs(dh_error_val_nn)):.2f} | {dh_inter_95_nn[0]:.2f}~{dh_inter_95_nn[1]:.2f} |\n'
    output_string_dw += f'| Neural Network model | {train_rmse_dw_nn:.2f} | {val_rmse_dw_nn:.2f} | {np.max(np.abs(dw_error_train_nn)):.2f} | {np.max(np.abs(dw_error_val_nn)):.2f} | {dw_inter_95_nn[0]:.2f}~{dw_inter_95_nn[1]:.2f} |\n'
    # Gaussian Process model. with torch height
    print("Training Gaussian Process model with torch height...")
    dh_error_train_gp, dw_error_train_gp, dh_error_val_gp, dw_error_val_gp = \
        train_GP(deepcopy(train_input), deepcopy(train_output), deepcopy(val_input), deepcopy(val_output), torch_height=True, layer_height=False, train_model=False, model_dir='weld_GP_models/')
    train_rmse_dh_gp, train_rmse_dw_gp, val_rmse_dh_gp, val_rmse_dw_gp = \
        get_rmse(dh_error_train_gp), get_rmse(dw_error_train_gp), get_rmse(dh_error_val_gp), get_rmse(dw_error_val_gp)
    dh_inter_95_gp, dw_inter_95_gp = plot_error_distribution(np.abs(dh_error_train_gp), np.abs(dw_error_train_gp), np.abs(dh_error_val_gp), np.abs(dw_error_val_gp), plot_title='Error distribution (GP model with torch height)') # plot training and validation error distribution for dh and dw
    plot_error_heatmap(dh_error_train_gp, dw_error_train_gp, dh_error_val_gp, dw_error_val_gp, train_input, val_input, plot_title='Error heatmap (GP model with torch height)') # plot training and validation error heatmap for dh and dw
    output_string_dh += f'| Gaussian Process model | {train_rmse_dh_gp:.2f} | {val_rmse_dh_gp:.2f} | {np.max(np.abs(dh_error_train_gp)):.2f} | {np.max(np.abs(dh_error_val_gp)):.2f} | {dh_inter_95_gp[0]:.2f}~{dh_inter_95_gp[1]:.2f} |\n'
    output_string_dw += f'| Gaussian Process model | {train_rmse_dw_gp:.2f} | {val_rmse_dw_gp:.2f} | {np.max(np.abs(dw_error_train_gp)):.2f} | {np.max(np.abs(dw_error_val_gp)):.2f} | {dw_inter_95_gp[0]:.2f}~{dw_inter_95_gp[1]:.2f} |\n'

    dh_error_all = {}
    dh_error_all['Linear loglog'] = np.abs(np.append(dh_error_train_lnln_lin, dh_error_val_lnln_lin))
    dh_error_all['Quadratic loglog'] = np.abs(np.append(dh_error_train_lnln_qua, dh_error_val_lnln_qua))
    dh_error_all['Neural Network'] = np.abs(np.append(dh_error_train_nn, dh_error_val_nn))
    dh_error_all['Gaussian Process'] = np.abs(np.append(dh_error_train_gp, dh_error_val_gp))
    dw_error_all = {}
    dw_error_all['Linear loglog'] = np.abs(np.append(dw_error_train_lnln_lin, dw_error_val_lnln_lin))
    dw_error_all['Quadratic loglog'] = np.abs(np.append(dw_error_train_lnln_qua, dw_error_val_lnln_qua))
    dw_error_all['Neural Network'] = np.abs(np.append(dw_error_train_nn, dw_error_val_nn))
    dw_error_all['Gaussian Process'] = np.abs(np.append(dw_error_train_gp, dw_error_val_gp))
    plot_error_distribution_all(dh_error_all, dw_error_all, plot_title='Error distribution (all models)') # plot training and validation error distribution for dh and dw


    print(output_string_dh)
    print(output_string_dw)

    ### NN ablation study
    # dh_error_train_nn, dw_error_train_nn, dh_error_val_nn, dw_error_val_nn = \
    #     train_NN(deepcopy(train_input), deepcopy(train_output), deepcopy(val_input), deepcopy(val_output), torch_height=False, layer_height=False, train_model=True, model_dir='weld_NN_models/model_vt_vw_')
    # train_rmse_dh_nn, train_rmse_dw_nn, val_rmse_dh_nn, val_rmse_dw_nn = \
    #     get_rmse(dh_error_train_nn), get_rmse(dw_error_train_nn), get_rmse(dh_error_val_nn), get_rmse(dw_error_val_nn)
    # dh_inter_95_nn, dw_inter_95_nn = plot_error_distribution(np.abs(dh_error_train_nn), np.abs(dw_error_train_nn), np.abs(dh_error_val_nn), np.abs(dw_error_val_nn), plot_title='Error distribution (neural network model)') # plot training and validation error distribution for dh and dw
    # plot_error_heatmap(dh_error_train_nn, dw_error_train_nn, dh_error_val_nn, dw_error_val_nn, train_input, val_input, plot_title='Error heatmap (neural network model)') # plot training and validation error heatmap for dh and dw
    # output_string_dh += f'| NN (v $\omega$) | {train_rmse_dh_nn:.2f} | {val_rmse_dh_nn:.2f} | {np.max(np.abs(dh_error_train_nn)):.2f} | {np.max(np.abs(dh_error_val_nn)):.2f} | {dh_inter_95_nn[0]:.2f}~{dh_inter_95_nn[1]:.2f} |\n'
    # output_string_dw += f'| NN (v $\omega$) | {train_rmse_dw_nn:.2f} | {val_rmse_dw_nn:.2f} | {np.max(np.abs(dw_error_train_nn)):.2f} | {np.max(np.abs(dw_error_val_nn)):.2f} | {dw_inter_95_nn[0]:.2f}~{dw_inter_95_nn[1]:.2f} |\n'
    
    # dh_error_train_nn, dw_error_train_nn, dh_error_val_nn, dw_error_val_nn = \
    #     train_NN(deepcopy(train_input), deepcopy(train_output), deepcopy(val_input), deepcopy(val_output), torch_height=True, layer_height=False, train_model=True, model_dir='weld_NN_models/model_vt_vw_ht_')
    # train_rmse_dh_nn, train_rmse_dw_nn, val_rmse_dh_nn, val_rmse_dw_nn = \
    #     get_rmse(dh_error_train_nn), get_rmse(dw_error_train_nn), get_rmse(dh_error_val_nn), get_rmse(dw_error_val_nn)
    # dh_inter_95_nn, dw_inter_95_nn = plot_error_distribution(np.abs(dh_error_train_nn), np.abs(dw_error_train_nn), np.abs(dh_error_val_nn), np.abs(dw_error_val_nn), plot_title='Error distribution (neural network model)') # plot training and validation error distribution for dh and dw
    # plot_error_heatmap(dh_error_train_nn, dw_error_train_nn, dh_error_val_nn, dw_error_val_nn, train_input, val_input, plot_title='Error heatmap (neural network model)') # plot training and validation error heatmap for dh and dw
    # output_string_dh += f'| NN (v $\omega$ $h_t$) | {train_rmse_dh_nn:.2f} | {val_rmse_dh_nn:.2f} | {np.max(np.abs(dh_error_train_nn)):.2f} | {np.max(np.abs(dh_error_val_nn)):.2f} | {dh_inter_95_nn[0]:.2f}~{dh_inter_95_nn[1]:.2f} |\n'
    # output_string_dw += f'| NN (v $\omega$ $h_t$) | {train_rmse_dw_nn:.2f} | {val_rmse_dw_nn:.2f} | {np.max(np.abs(dw_error_train_nn)):.2f} | {np.max(np.abs(dw_error_val_nn)):.2f} | {dw_inter_95_nn[0]:.2f}~{dw_inter_95_nn[1]:.2f} |\n'
    
    # dh_error_train_nn, dw_error_train_nn, dh_error_val_nn, dw_error_val_nn = \
    #     train_NN(deepcopy(train_input), deepcopy(train_output), deepcopy(val_input), deepcopy(val_output), torch_height=False, layer_height=True, train_model=True, model_dir='weld_NN_models/model_vt_vw_hl_')
    # train_rmse_dh_nn, train_rmse_dw_nn, val_rmse_dh_nn, val_rmse_dw_nn = \
    #     get_rmse(dh_error_train_nn), get_rmse(dw_error_train_nn), get_rmse(dh_error_val_nn), get_rmse(dw_error_val_nn)
    # dh_inter_95_nn, dw_inter_95_nn = plot_error_distribution(np.abs(dh_error_train_nn), np.abs(dw_error_train_nn), np.abs(dh_error_val_nn), np.abs(dw_error_val_nn), plot_title='Error distribution (neural network model)') # plot training and validation error distribution for dh and dw
    # plot_error_heatmap(dh_error_train_nn, dw_error_train_nn, dh_error_val_nn, dw_error_val_nn, train_input, val_input, plot_title='Error heatmap (neural network model)') # plot training and validation error heatmap for dh and dw
    # output_string_dh += f'| NN (v $\omega$ $h_l$) | {train_rmse_dh_nn:.2f} | {val_rmse_dh_nn:.2f} | {np.max(np.abs(dh_error_train_nn)):.2f} | {np.max(np.abs(dh_error_val_nn)):.2f} | {dh_inter_95_nn[0]:.2f}~{dh_inter_95_nn[1]:.2f} |\n'
    # output_string_dw += f'| NN (v $\omega$ $h_l$) | {train_rmse_dw_nn:.2f} | {val_rmse_dw_nn:.2f} | {np.max(np.abs(dw_error_train_nn)):.2f} | {np.max(np.abs(dw_error_val_nn)):.2f} | {dw_inter_95_nn[0]:.2f}~{dw_inter_95_nn[1]:.2f} |\n'
    
    # dh_error_train_nn, dw_error_train_nn, dh_error_val_nn, dw_error_val_nn = \
    #     train_NN(deepcopy(train_input), deepcopy(train_output), deepcopy(val_input), deepcopy(val_output), torch_height=True, layer_height=True, train_model=True, model_dir='weld_NN_models/model_vt_vw_ht_hl_')
    # train_rmse_dh_nn, train_rmse_dw_nn, val_rmse_dh_nn, val_rmse_dw_nn = \
    #     get_rmse(dh_error_train_nn), get_rmse(dw_error_train_nn), get_rmse(dh_error_val_nn), get_rmse(dw_error_val_nn)
    # dh_inter_95_nn, dw_inter_95_nn = plot_error_distribution(np.abs(dh_error_train_nn), np.abs(dw_error_train_nn), np.abs(dh_error_val_nn), np.abs(dw_error_val_nn), plot_title='Error distribution (neural network model)') # plot training and validation error distribution for dh and dw
    # plot_error_heatmap(dh_error_train_nn, dw_error_train_nn, dh_error_val_nn, dw_error_val_nn, train_input, val_input, plot_title='Error heatmap (neural network model)') # plot training and validation error heatmap for dh and dw
    # output_string_dh += f'| NN (v $\omega$ $h_t$ $h_l$) | {train_rmse_dh_nn:.2f} | {val_rmse_dh_nn:.2f} | {np.max(np.abs(dh_error_train_nn)):.2f} | {np.max(np.abs(dh_error_val_nn)):.2f} | {dh_inter_95_nn[0]:.2f}~{dh_inter_95_nn[1]:.2f} |\n'
    # output_string_dw += f'| NN (v $\omega$ $h_t$ $h_l$) | {train_rmse_dw_nn:.2f} | {val_rmse_dw_nn:.2f} | {np.max(np.abs(dw_error_train_nn)):.2f} | {np.max(np.abs(dw_error_val_nn)):.2f} | {dw_inter_95_nn[0]:.2f}~{dw_inter_95_nn[1]:.2f} |\n'
    
    # print(output_string_dh)
    # print(output_string_dw)

if __name__ == "__main__":
    main()