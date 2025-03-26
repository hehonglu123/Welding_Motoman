import numpy as np
import pandas as pd
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

# static randomize seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

def plot_vt_vw_dh_dw(v_torch_range, v_wire_range, dh_pred, dw_pred, plot_title='Predicted dh and dw'):

    # make the v_torch_range and v_wire_range 2D arrays for imshow
    v_torch_range_grid = np.repeat(v_torch_range, len(v_wire_range)).reshape(len(v_torch_range), len(v_wire_range))
    v_wire_range_grid = np.tile(v_wire_range, len(v_torch_range)).reshape(len(v_torch_range), len(v_wire_range))
    v_wire_range_ipm = v_wire_range * mm2inch * 60 # convert mm/s to ipm

    xy_label_size = 14
    xy_tick_size = 12
    title_size = 16
    sup_title_size = 18
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

def plot_error_distribution(dh_train_error, dw_train_error, dh_val_error, dw_val_error):

    # fit distribution with exponential distribution

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # dh train error
    ax[0].hist(dh_train_error, bins=50, alpha=0.5, label='Train')
    # dh validation error
    ax[0].hist(dh_val_error, bins=50, alpha=0.5, label='Test')
    ax[0].set_xlabel('dh Error (mm)')
    ax[0].set_ylabel('Count')
    ax[0].set_title('dh Error Distribution')
    ax[0].legend()
    ax[0].grid()
    # dw train error 
    ax[1].hist(dw_train_error, bins=50, alpha=0.5, label='Train')
    # dw validation error
    ax[1].hist(dw_val_error, bins=50, alpha=0.5, label='Test')
    ax[1].set_xlabel('dw Error (mm)')
    ax[1].set_ylabel('Count')
    ax[1].set_title('dw Error Distribution')
    ax[1].legend()
    ax[1].grid()
    plt.tight_layout()
    plt.show()

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
    theta_param_dh = np.linalg.pinv(mat_A_train)@mat_B_dh
    theta_param_dw = np.linalg.pinv(mat_A_train)@mat_B_dw

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
    
    train_rmse_dh = np.sqrt(np.mean(train_dh_error**2))
    train_rmse_dw = np.sqrt(np.mean(train_dw_error**2))
    val_rmse_dh = np.sqrt(np.mean(val_dh_error**2))
    val_rmse_dw = np.sqrt(np.mean(val_dw_error**2))

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
    plot_title = 'Predicted dh and dw using log-log linear model' if not quadratic else 'Predicted dh and dw using log-log quadratic model'
    plot_vt_vw_dh_dw(v_torch_range, v_wire_range, dh_pred, dw_pred, plot_title=plot_title)
    
    return train_rmse_dh, train_rmse_dw, val_rmse_dh, val_rmse_dw, train_dh_error, train_dw_error, val_dh_error, val_dw_error
    
def train_NN(train_input,train_output,val_input,val_output,torch_height=True,layer_height=False):

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
    learning_rate = 0.001
    # Define the number of epochs
    num_epochs = 3000
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ### training loop ###
    training_loss_all = []
    validation_loss_all = []
    for epoch in range(num_epochs):
        # Forward pass
        train_output_pred = model(train_input)
        # Compute the loss
        loss = loss_fn(train_output_pred, train_output)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        val_output_pred = model(val_input)
        val_loss = loss_fn(val_output_pred, val_output)

        # Store the losses for plotting
        training_loss_all.append(loss.item())
        validation_loss_all.append(val_loss.item())

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')
    # final training loss after last epoch
    train_output_pred = model(train_input)
    val_output_pred = model(val_input)
    loss = loss_fn(train_output_pred, train_output)
    val_loss = loss_fn(val_output_pred, val_output)
    print("Final Training Loss: ", loss.item())
    print(type(loss.item()))
    print("Final Validation Loss: ", val_loss.item())
    train_rmse_dh = loss_fn(train_output_pred[:,0], train_output[:,0]).item()
    train_rmse_dw = loss_fn(train_output_pred[:,1], train_output[:,1]).item()
    val_rmse_dh = loss_fn(val_output_pred[:,0], val_output[:,0]).item()
    val_rmse_dw = loss_fn(val_output_pred[:,1], val_output[:,1]).item()
    dh_error_train = train_output_pred[:,0].detach().numpy() - train_output[:,0].detach().numpy()
    dw_error_train = train_output_pred[:,1].detach().numpy() - train_output[:,1].detach().numpy()
    dh_error_val = val_output_pred[:,0].detach().numpy() - val_output[:,0].detach().numpy()
    dw_error_val = val_output_pred[:,1].detach().numpy() - val_output[:,1].detach().numpy()

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
    plot_vt_vw_dh_dw(v_torch_range, v_wire_range, dh_pred, dw_pred, plot_title='Predicted dh and dw using Neural Network model')

    return train_rmse_dh, train_rmse_dw, val_rmse_dh, val_rmse_dw, dh_error_train, dw_error_train, dh_error_val, dw_error_val

def main():
    
    # read data from weld_data
    data_dir = '../../data/wall_weld_test/'
    weld_data = pd.read_csv(data_dir + 'weld_data.csv', header=0)
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
    output_string_dh += '| | Train RMSE dh| Test RMSE dh | Train Max dh | Test Max dh |\n'
    output_string_dh += '|---|---|---|---|---|\n'

    output_string_dw = '# Error Distribution\n'
    output_string_dw += '| | Train RMSE dw| Test RMSE dw | Train Max dw | Test Max dw |\n'
    output_string_dw += '|---|---|---|---|---|\n'

    # log-log model. linear
    print("Training log-log linear model...")
    train_rmse_dh_lnln_lin, train_rmse_dw_lnln_lin, val_rmse_dh_lnln_lin, val_rmse_dw_lnln_lin, dh_error_train_lnln_lin, dw_error_train_lnln_lin, dh_error_val_lnln_lin, dw_error_val_lnln_lin = \
        train_loglog(deepcopy(train_input), deepcopy(train_output), deepcopy(val_input), deepcopy(val_output),quadratic=False)
    plot_error_distribution(dh_error_train_lnln_lin, dw_error_train_lnln_lin, dh_error_val_lnln_lin, dw_error_val_lnln_lin) # plot training and validation error distribution for dh and dw
    output_string_dh += f'| Linear loglog model | {train_rmse_dh_lnln_lin:.2f} | {val_rmse_dh_lnln_lin:.2f} | {np.max(np.abs(dh_error_train_lnln_lin)):.2f} | {np.max(np.abs(dh_error_val_lnln_lin)):.2f} |\n'
    output_string_dw += f'| Linear loglog model | {train_rmse_dw_lnln_lin:.2f} | {val_rmse_dw_lnln_lin:.2f} | {np.max(np.abs(dw_error_train_lnln_lin)):.2f} | {np.max(np.abs(dw_error_val_lnln_lin)):.2f} |\n'
    # log-log model. with quadratic term
    print("Training log-log quadratic model...")
    train_rmse_dh_lnln_qua, train_rmse_dw_lnln_qua, val_rmse_dh_lnln_qua, val_rmse_dw_lnln_qua, dh_error_train_lnln_qua, dw_error_train_lnln_qua, dh_error_val_lnln_qua, dw_error_val_lnln_qua = \
        train_loglog(deepcopy(train_input), deepcopy(train_output), deepcopy(val_input), deepcopy(val_output),quadratic=True)
    plot_error_distribution(dh_error_train_lnln_qua, dw_error_train_lnln_qua, dh_error_val_lnln_qua, dw_error_val_lnln_qua) # plot training and validation error distribution for dh and dw
    output_string_dh += f'| Quadratic loglog model | {train_rmse_dh_lnln_qua:.2f} | {val_rmse_dh_lnln_qua:.2f} | {np.max(np.abs(dh_error_train_lnln_qua)):.2f} | {np.max(np.abs(dh_error_val_lnln_qua)):.2f} |\n'
    output_string_dw += f'| Quadratic loglog model | {train_rmse_dw_lnln_qua:.2f} | {val_rmse_dw_lnln_qua:.2f} | {np.max(np.abs(dw_error_train_lnln_qua)):.2f} | {np.max(np.abs(dw_error_val_lnln_qua)):.2f} |\n'
    # Neural Network model. with torch height
    print("Training Neural Network model with torch height...")
    train_rmse_dh_nn, train_rmse_dw_nn, val_rmse_dh_nn, val_rmse_dw_nn, dh_error_train_nn, dw_error_train_nn, dh_error_val_nn, dw_error_val_nn = \
        train_NN(deepcopy(train_input), deepcopy(train_output), deepcopy(val_input), deepcopy(val_output))
    plot_error_distribution(dh_error_train_nn, dw_error_train_nn, dh_error_val_nn, dw_error_val_nn) # plot training and validation error distribution for dh and dw
    output_string_dh += f'| Neural Network model | {train_rmse_dh_nn:.2f} | {val_rmse_dh_nn:.2f} | {np.max(np.abs(dh_error_train_nn)):.2f} | {np.max(np.abs(dh_error_val_nn)):.2f} |\n'
    output_string_dw += f'| Neural Network model | {train_rmse_dw_nn:.2f} | {val_rmse_dw_nn:.2f} | {np.max(np.abs(dw_error_train_nn)):.2f} | {np.max(np.abs(dw_error_val_nn)):.2f} |\n'

    print(output_string_dh)
    print(output_string_dw)

if __name__ == "__main__":
    main()