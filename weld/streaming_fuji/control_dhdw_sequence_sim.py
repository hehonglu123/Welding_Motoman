import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import sys, datetime, yaml, pathlib, glob, os, time, argparse
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

# for neural network input normalization
all_inputs = np.loadtxt('weld_Seq_models/test_cmd_v_feedrate.csv', delimiter=',',skiprows=1)
all_inputs = np.vstack((all_inputs, np.loadtxt('weld_Seq_models/train_cmd_v_feedrate.csv', delimiter=',',skiprows=1)))
v_max = np.max(all_inputs[:, 0])
v_min = np.min(all_inputs[:, 0])
feedrate_max = np.max(all_inputs[:, 1])
feedrate_min = np.min(all_inputs[:, 1])
print(f"v_min: {v_min}, v_max: {v_max}, feedrate_min: {feedrate_min}, feedrate_max: {feedrate_max}")
feedrate_delta = 10 / (feedrate_max-feedrate_min) # 10 is discretized feedrate value

np.random.seed(42) # for reproducibility
torch.manual_seed(42) # for reproducibility

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

def load_model(model_dir):

    model_dir = 'weld_Seq_models/'+model_dir+'/'
    # load the parameters
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
    try:
        use_stickout_length = training_params['use_stickout_length']
    except KeyError:
        use_stickout_length = False
    try:
        latency = training_params['latency']
    except KeyError:
        latency = 0
    latency_steps = int(latency * sample_rate) # number of steps to consider for latency

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
        model = modelClass(input_size=model_input_size, hidden_size=model_hidden_size, output_size=model_output_size, num_layers=num_layers, history_length=history_length, latency_steps=latency_steps, open_loop=open_loop, device=device).to(device)

    # load the model state
    model_state_dict = torch.load(model_dir + 'best_model.pth', map_location=device, weights_only=True)
    model.load_state_dict(model_state_dict)

    return model, training_params

def normalize_input(u):
    """
    Normalize the input based on the predefined min and max values.
    """
    u_in = deepcopy(u)
    u_in[:, 0] = (u_in[:, 0] - v_min) / (v_max - v_min)
    u_in[:, 1] = (u_in[:, 1] - feedrate_min) / (feedrate_max - feedrate_min)
    return u_in

def denormalize_input(u_normed):
    """
    Denormalize the input based on the predefined min and max values.
    """
    u = np.zeros_like(u_normed)
    u[:, 0] = u_normed[:, 0] * (v_max - v_min) + v_min
    u[:, 1] = u_normed[:, 1] * (feedrate_max - feedrate_min) + feedrate_min
    return u

def mixed_input_correction(J:torch.tensor, delta_y:torch.tensor, u_cont_prev:torch.tensor, u_disc_prev:torch.tensor,
                           model, h_t, alpha, y_desired,
                            disc_search_radius=feedrate_delta, lambda_smooth=1e-2, lambda_disc=1e-2):
    """
    J: Jacobian matrix, shape (output_dim, input_dim)
    delta_y: Desired output change, shape (output_dim, 1)
    u_cont_prev: previous continuous input (torch speed) tensor of shape (cont_dim,)
    u_disc_prev: previous discrete input (wire feed rate) scalar

    Returns: Updated (u_cont_new, u_disc_new)
    """
    input_dim = J.shape[1]
    cont_dim = u_cont_prev.shape[0]
    disc_dim = 1

    J_cont = J[:, :cont_dim]  # (output_dim, cont_dim)
    J_disc = J[:, cont_dim:]  # (output_dim, disc_dim)

    # JT = J.T
    # lhs = JT @ J + torch.diag(torch.tensor([lambda_smooth, lambda_disc],device=device))@torch.eye(J.shape[1], device=device)
    # rhs = JT @ delta_y
    # delta_u = torch.linalg.solve(lhs, rhs).T
    # return delta_u

    best_cost = float('inf')
    best_u_disc = None
    best_u_cont = None

    for delta_disc_index in range(-1,2):
        delta_disc = delta_disc_index * disc_search_radius
        u_disc_candidate = u_disc_prev + delta_disc

        # Compute rhs of least squares
        rhs = delta_y - J_disc @ torch.tensor([[delta_disc]], dtype=J.dtype, device=J.device)

        JTJ = J_cont.T @ J_cont + lambda_smooth * torch.eye(cont_dim, device=J.device)
        JTr = J_cont.T @ rhs

        delta_u_cont = torch.linalg.solve(JTJ, JTr)

        # cost using the linearized model
        residual = J_cont @ delta_u_cont + J_disc @ torch.tensor([[delta_disc]], dtype=J.dtype, device=J.device) - delta_y
        cost = residual.norm()**2 + lambda_smooth * (delta_u_cont.norm()**2) + lambda_disc * (delta_disc**2)
        # cost using the original model
        # u_candidate = torch.stack([u_cont_prev[0] + delta_u_cont.view(-1), u_disc_candidate], dim=1)
        # y_pred_control_t, _ = model.forward_one_step(torch.cat((u_candidate, torch.zeros((1,2),device=device)), dim=1), h_t.clone().detach())
        # residual = y_desired - y_pred_control_t
        # cost = residual.norm()**2 + lambda_smooth * (delta_u_cont.norm()**2) + lambda_disc * (delta_disc**2)

        if cost < best_cost:
            best_cost = cost
            best_u_disc = u_disc_candidate
            best_u_cont = u_cont_prev + delta_u_cont.view(-1)

    return best_u_cont, best_u_disc

def main():

    model_dir = 'model_20250715_151650'

    model_control, model_params_control = load_model(model_dir)
    model_sim, model_params_sim = load_model(model_dir)
    model_control.eval()
    model_sim.eval()

    lambda_smooth = 1e-2  # regularization parameter for smoothness
    lambda_disc = 1e-1*5  # regularization parameter for discrete input
    alpha = 0.05 # step size for continuous input correction
    
    d_dh = [2,1.8,2.2,2.4,1.6] # mm
    d_width = [5,4.4,4.6,5.2,4] # mm
    sim_steps = 500
    y_target = np.vstack((d_dh, d_width)).T  # target values for height and width
    y_target = torch.tensor(y_target, dtype=torch.float32, device=device)
    step_break = [0, 100, 200, 300, 400]  # breakpoints for step changes in target values
    assert len(step_break) == len(d_dh) == len(d_width), "Step breakpoints must match target values length"

    # initial input: [velocity, feedrate]
    # random guess or from a log-log model
    u_init = np.array([[5.,170.]], dtype=np.float64)  # initial guess for velocity and feedrate
    print("Normalizing input:", normalize_input(u_init))
    print("Denormalizing input:", denormalize_input(normalize_input(u_init)))

    u_t = torch.tensor(normalize_input(u_init), dtype=torch.float32, device=device)
    u_t_sim = u_t.clone().detach()  # for simulation model
    h_t = torch.zeros((1, model_params_control['model_hidden_size']), device=device)
    h_t_sim = h_t.clone().detach()  # for simulation model
    output_y_seq = []
    input_u_seq = []
    for t_step in range(sim_steps):
        target_index = np.searchsorted(step_break, t_step, side='right') - 1

        # feedforward to the simulation model
        u_t_sim = u_t.clone().detach()
        y_pred_sim_t, h_t_sim = model_sim.forward_one_step(torch.cat((u_t_sim, torch.zeros_like(y_target[target_index:target_index+1])), dim=1), h_t_sim)
        output_y_seq.append(y_pred_sim_t)

        # feedforward to the control model, to obtrain the Jacobian
        input_u_seq.append(u_t.detach().cpu().numpy()[0])
        u_t = u_t.clone().detach().requires_grad_(True)  # ensure u_t is differentiable
        y_pred_control_t, h_t = model_control.forward_one_step(torch.cat((u_t, torch.zeros_like(y_target[target_index:target_index+1])), dim=1), h_t)
        # Compute Jacobian dy/du
        jacobian = []
        for i in range(y_pred_control_t.size(1)):
            grad = torch.autograd.grad(y_pred_control_t[0, i], u_t, retain_graph=True)[0]
            jacobian.append(grad[0].detach())
        J = torch.stack(jacobian, dim=0)  # Shape: (output_dim, input_dim)

        # Compute the desired y
        delta_y_desired = (y_target[target_index] - y_pred_sim_t).detach().T

        # Solve mixed input correction
        # delta_u = mixed_input_correction(
        #     J, delta_y_desired, u_t[:, 0], u_t[:, 1],
        #     model_control, h_t, alpha, y_target[target_index],
        #     disc_search_radius=feedrate_delta,
        #     lambda_smooth=lambda_smooth, lambda_disc=lambda_disc
        # )
        # u_t = u_t + alpha * delta_u  # Update u_t with the correction
        # u_t[:,1] = torch.round(u_t[:, 1] / feedrate_delta) * feedrate_delta  # Discretize the feedrate input

        u_cont_new, u_disc_new = mixed_input_correction(
            J, delta_y_desired, u_t[:, 0], u_t[:, 1],
            model_control, h_t, alpha, y_target[target_index],
            disc_search_radius=feedrate_delta,
            lambda_smooth=lambda_smooth, lambda_disc=lambda_disc
        )
        u_t_cont_new = u_t[:, 0] + alpha * (u_cont_new - u_t[:, 0])
        u_t = torch.stack([u_t_cont_new, u_disc_new], dim=1)
    
    # plot the results output_y_seq and y target vs time
    output_y_seq = torch.stack(output_y_seq, dim=0).detach().cpu().numpy()
    input_u_seq = np.array(input_u_seq)
    input_u_seq = denormalize_input(input_u_seq)
    time_elapse = np.arange(sim_steps)/model_params_control['sample_rate']
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.plot(time_elapse, output_y_seq[:, 0, 0], label=f'Simulated $\Delta h$')
    for step_i in range(len(step_break)):
        if step_i < len(step_break) - 1:
            plt.plot(time_elapse[step_break[step_i]:step_break[step_i+1]], y_target[step_i, 0].item() * np.ones(step_break[step_i+1] - step_break[step_i]), 'r--', label=f'Target $\Delta h$' if step_i == 0 else "")
        else:
            plt.plot(time_elapse[step_break[step_i]:], y_target[step_i, 0].item() * np.ones(len(time_elapse) - step_break[step_i]), 'r--', label=f'Target $\Delta h$' if step_i == 0 else "")
    plt.xlabel('Time (s)', fontsize=xy_label_size)
    plt.ylabel(f'$\Delta h$ (mm)', fontsize=xy_label_size)
    plt.legend(fontsize=xy_label_size)
    plt.title('Height Control', fontsize=xy_label_size)
    plt.grid()
    plt.subplot(2, 2, 2)
    plt.plot(time_elapse, output_y_seq[:, 0, 1], label=f'Simulated $w$', color='orange')
    for step_i in range(len(step_break)):
        if step_i < len(step_break) - 1:
            plt.plot(time_elapse[step_break[step_i]:step_break[step_i+1]], y_target[step_i, 1].item() * np.ones(step_break[step_i+1] - step_break[step_i]), 'r--', label=f'Target $w$' if step_i == 0 else "")
        else:
            plt.plot(time_elapse[step_break[step_i]:], y_target[step_i, 1].item() * np.ones(len(time_elapse) - step_break[step_i]), 'r--', label=f'Target $w$' if step_i == 0 else "")

    plt.xlabel('Time (s)', fontsize=xy_label_size)
    plt.ylabel(f'$w$ (mm)', fontsize=xy_label_size)
    plt.legend(fontsize=xy_label_size)
    plt.title('Width Control', fontsize=xy_label_size)
    plt.grid()
    # plt.tight_layout()
    plt.subplot(2, 2, 3)
    plt.plot(time_elapse, input_u_seq[:, 0], label='Velocity (mm/s)')
    plt.xlabel('Time (s)', fontsize=xy_label_size)
    plt.ylabel('Velocity (mm/s)', fontsize=xy_label_size)
    plt.legend(fontsize=xy_label_size)
    plt.title('Input Velocity', fontsize=xy_label_size)
    plt.grid()
    plt.subplot(2, 2, 4)
    plt.plot(time_elapse, input_u_seq[:, 1], label='Feedrate (inch/min)', color='orange')
    plt.xlabel('Time (s)', fontsize=xy_label_size)
    plt.ylabel('Feedrate (inch/min)', fontsize=xy_label_size)
    plt.legend(fontsize=xy_label_size)
    plt.title('Input Feedrate', fontsize=xy_label_size)
    plt.grid()
    # plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()