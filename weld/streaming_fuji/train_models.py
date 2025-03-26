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

def train_loglog(data):

    input_keys = ['v_torch','v_wire']
    output_keys = ['dh','dw']

    # data_input = np.vstack((np.log(data['v_torch'])*np.log(data['v_wire']), np.log(data['v_torch']), np.log(data['v_wire']), np.ones_like(data['v_torch']))).T
    data_input = np.vstack((np.log(data['v_torch']), np.log(data['v_wire']), np.ones_like(data['v_torch']))).T
    dh_new_output = np.log(data['dh'])
    dw_new_output = np.log(data['dw'])

    theta_param_dh = np.linalg.pinv(data_input)@dh_new_output
    theta_param_dw = np.linalg.pinv(data_input)@dw_new_output

    rmse_dh = np.sqrt(np.mean((np.exp(dh_new_output)-np.exp(data_input@theta_param_dh))**2))
    rmse_dw = np.sqrt(np.mean((np.exp(dw_new_output)-np.exp(data_input@theta_param_dw))**2))
    max_dh_error = np.max(np.abs(np.exp(dh_new_output)-np.exp(data_input@theta_param_dh)))
    max_dw_error = np.max(np.abs(np.exp(dw_new_output)-np.exp(data_input@theta_param_dw)))

    print("Linear model")
    print("dh rmse: ", rmse_dh)
    print("dw rmse: ", rmse_dw)
    print("dh max error: ", max_dh_error)
    print("dw max error: ", max_dw_error)
    

def train_NN(data):

    # load data to numpy arrays
    # input_keys = ['height','v_torch','v_wire','torch_height']
    # input_keys = ['height','v_torch','v_wire']
    input_keys = ['torch_height','v_torch','v_wire']
    # input_keys = ['v_torch','v_wire']
    output_keys = ['dh','dw']
    train_input = []
    train_output = []
    for key in input_keys:
        train_input.append(np.array(data[key]))
    for key in output_keys:
        train_output.append(np.array(data[key]))
    train_input = np.array(train_input).T
    train_output = np.array(train_output).T

    # convert to torch tensors
    train_input = torch.tensor(train_input, dtype=torch.float32)
    train_output = torch.tensor(train_output, dtype=torch.float32)

    # choose 20% of the data for validation
    training_percent = 0.8
    num_samples = train_input.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split = int(np.floor(training_percent * num_samples))
    train_indices, val_indices = indices[:split], indices[split:]
    train_input, val_input = train_input[train_indices], train_input[val_indices]
    train_output, val_output = train_output[train_indices], train_output[val_indices]

    # Define the input size, hidden size, and output size
    input_size = train_input.shape[1]
    hidden_sizes = [200,200,200]
    output_size = train_output.shape[1]
    # Define the model
    model = NeuralNetwork(input_size, output_size, hidden_sizes=hidden_sizes)
    # Define loss (mean squared error) and optimizer (Adam)
    loss_fn = nn.MSELoss()
    learning_rate = 0.001
    # Define the number of epochs
    num_epochs = 1000
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
    print("Final Training Loss: ", loss.item())
    print("Final Validation Loss: ", val_loss.item())
    print("Final dh training loss: ", loss_fn(train_output_pred[:,0], train_output[:,0]).item())
    print("Final dw training loss: ", loss_fn(train_output_pred[:,1], train_output[:,1]).item())
    print("Final dh validation loss: ", loss_fn(val_output_pred[:,0], val_output[:,0]).item())
    print("Final dw validation loss: ", loss_fn(val_output_pred[:,1], val_output[:,1]).item())
    # max error
    dh_max_error_train = np.max(np.abs(train_output_pred[:,0].detach().numpy() - train_output[:,0].detach().numpy()))
    dh_max_error_train_v_torch = train_input.detach().numpy()[:,0][np.argmax(np.abs(train_output_pred[:,0].detach().numpy() - train_output[:,0].detach().numpy()))]
    dh_max_error_train_v_wire = train_input.detach().numpy()[:,1][np.argmax(np.abs(train_output_pred[:,0].detach().numpy() - train_output[:,0].detach().numpy()))]
    dw_max_error_train = np.max(np.abs(train_output_pred[:,1].detach().numpy() - train_output[:,1].detach().numpy()))
    dw_max_error_train_v_torch = train_input.detach().numpy()[:,0][np.argmax(np.abs(train_output_pred[:,1].detach().numpy() - train_output[:,1].detach().numpy()))]
    dw_max_error_train_v_wire = train_input.detach().numpy()[:,1][np.argmax(np.abs(train_output_pred[:,1].detach().numpy() - train_output[:,1].detach().numpy()))]
    dh_max_error_val = np.max(np.abs(val_output_pred[:,0].detach().numpy() - val_output[:,0].detach().numpy()))
    dh_max_error_val_v_torch = val_input.detach().numpy()[:,0][np.argmax(np.abs(val_output_pred[:,0].detach().numpy() - val_output[:,0].detach().numpy()))]
    dh_max_error_val_v_wire = val_input.detach().numpy()[:,1][np.argmax(np.abs(val_output_pred[:,0].detach().numpy() - val_output[:,0].detach().numpy()))]
    dw_max_error_val = np.max(np.abs(val_output_pred[:,1].detach().numpy() - val_output[:,1].detach().numpy()))
    dw_max_error_val_v_torch = val_input.detach().numpy()[:,0][np.argmax(np.abs(val_output_pred[:,1].detach().numpy() - val_output[:,1].detach().numpy()))]
    dw_max_error_val_v_wire = val_input.detach().numpy()[:,1][np.argmax(np.abs(val_output_pred[:,1].detach().numpy() - val_output[:,1].detach().numpy()))]
    print("Max error dh train: ", dh_max_error_train, 'at v_torch:', dh_max_error_train_v_torch, 'v_wire:', dh_max_error_train_v_wire*mm2inch*60)
    print("Max error dw train: ", dw_max_error_train, 'at v_torch:', dw_max_error_train_v_torch, 'v_wire:', dw_max_error_train_v_wire*mm2inch*60)
    print("Max error dh val: ", dh_max_error_val, 'at v_torch:', dh_max_error_val_v_torch, 'v_wire:', dh_max_error_val_v_wire*mm2inch*60)
    print("Max error dw val: ", dw_max_error_val, 'at v_torch:', dw_max_error_val_v_torch, 'v_wire:', dw_max_error_val_v_wire*mm2inch*60)
    exit()
    # dh dw at v_torch = 0.5, v_wire = 100 ipm
    v_torch = 2
    v_wire = 100 * inch2mm / 60 # ipm to mm/s
    input_data = torch.tensor(np.array([v_torch, v_wire]), dtype=torch.float32).unsqueeze(0) # add batch dimension
    with torch.no_grad():
        output_data = model(input_data)
    dh_pred, dw_pred = output_data.squeeze().numpy()
    print("Torch speed, wire feedrate:", v_torch, v_wire * mm2inch * 60, "ipm")
    print("Predicted dh: ", dh_pred)
    print("Predicted dw: ", dw_pred)

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
    dh_pred = np.zeros((len(v_torch_range), len(v_wire_range)))
    dw_pred = np.zeros((len(v_torch_range), len(v_wire_range)))
    for i, v_torch in enumerate(v_torch_range):
        for j, v_wire in enumerate(v_wire_range):
            input_data = torch.tensor(np.array([v_torch, v_wire]), dtype=torch.float32).unsqueeze(0) # add batch dimension
            with torch.no_grad():
                output_data = model(input_data)
            dh_pred[i,j], dw_pred[i,j] = output_data.squeeze().numpy()
    dh_pred = dh_pred.flatten()
    dw_pred = dw_pred.flatten()
    v_torch_range = np.repeat(v_torch_range, len(v_wire_range))
    v_wire_range = np.tile(v_wire_range, len(v_torch_range)//len(v_wire_range))
    # plot dh and dw using colormap and imshow
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # dh
    sc = ax[0].scatter(v_torch_range, v_wire_range, c=dh_pred, cmap='viridis', marker='o')
    ax[0].set_xlabel('Torch Speed (mm/s)')
    ax[0].set_ylabel('Wire Feedrate (ipm)')
    ax[0].set_yticks(np.arange(100, 201, 20) * inch2mm / 60)
    ax[0].set_yticklabels(np.arange(100, 201, 20))
    ax[0].set_title('Predicted dh')
    ax[0].set_aspect('auto')
    ax[0].grid()
    cbar = plt.colorbar(sc, ax=ax[0])
    cbar.set_label('dh (mm)')
    # dw
    sc = ax[1].scatter(v_torch_range, v_wire_range, c=dw_pred, cmap='viridis', marker='o')
    ax[1].set_xlabel('Torch Speed (mm/s)')
    ax[1].set_ylabel('Wire Feedrate (ipm)')
    ax[1].set_yticks(np.arange(100, 201, 20) * inch2mm / 60)
    ax[1].set_yticklabels(np.arange(100, 201, 20))
    ax[1].set_title('Predicted dw')
    ax[1].set_aspect('auto')
    ax[1].grid()
    cbar = plt.colorbar(sc, ax=ax[1])
    cbar.set_label('dw (mm)')
    plt.tight_layout()
    plt.show()
    

def main():
    
    # read data from weld_data
    data_dir = '../../data/wall_weld_test/'
    weld_data = pd.read_csv(data_dir + 'weld_data.csv', header=0)
    weld_data = weld_data.to_dict(orient='list')

    train_NN(data=weld_data)
    # train_loglog(data=weld_data)

if __name__ == "__main__":
    main()