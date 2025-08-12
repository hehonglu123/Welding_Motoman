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
from controlModelFunction import *

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

def main():

    model_dir = 'model_20250715_151650'

    crtlModel_control = controlModel(model_dir,device=device)
    model_control, model_params_control = crtlModel_control.get_model()
    crtlModel_sim = controlModel(model_dir,device=device)
    model_sim, model_params_sim = crtlModel_sim.get_model()
    crtlModel_control.model.eval()
    crtlModel_sim.model.eval()

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
    # u_init = np.array([[5.,170.]], dtype=np.float64)  # initial guess for velocity and feedrate
    u_init = np.array([[7.56,150.]], dtype=np.float64)  # initial guess for velocity and feedrate
    print("Normalizing input:", crtlModel_control.normalize_input(u_init))
    print("Denormalizing input:", crtlModel_control.denormalize_input(crtlModel_control.normalize_input(u_init)))

    

    u_t = torch.tensor(crtlModel_control.normalize_input(u_init), dtype=torch.float32, device=device)
    u_t_sim = u_t.clone().detach()  # for simulation model
    h_t = torch.zeros((1, model_params_control['model_hidden_size']), device=device)
    # h_t_sim = h_t.clone().detach()  # for simulation model
    # generate initial hidden state for simulation model using random number between -1 and 1
    h_t_sim = torch.rand((1, model_params_sim['model_hidden_size']), device=device) * 2 - 1  # random initialization between -1 and 1
    h_t_sim_seq = [h_t_sim.detach().cpu().numpy()[0]]
    h_t_seq = [h_t.detach().cpu().numpy()[0]]
    output_y_seq = []
    input_u_seq = []
    error_y_pred_seq = []

    # for t_step in range(sim_steps):
    #     target_index = np.searchsorted(step_break, t_step, side='right') - 1
    #     # feedforward to the simulation model
    #     u_t_sim = u_t.clone().detach()
    #     y_pred_sim_t, h_t_sim = crtlModel_sim.model.forward_one_step(torch.cat((u_t_sim, torch.zeros_like(y_target[target_index:target_index+1])), dim=1), h_t_sim)
    #     output_y_seq.append(y_pred_sim_t)
    #     h_t_sim_seq.append(h_t_sim.detach().cpu().numpy()[0])
    # output_y_seq = torch.stack(output_y_seq, dim=0).detach().cpu().numpy()
    # # plot dh and width vs time
    # time_elapse = np.arange(sim_steps)/model_params_control['sample_rate']
    # plt.figure(figsize=(12, 6))
    # plt.subplot(2, 1, 1)
    # plt.plot(time_elapse, output_y_seq[:, 0, 0], label=f'Simulated $\Delta h$')
    # plt.show()
    # exit()

    # the very first control model feedforward prediction
    # y_pred_control_t, h_t = crtlModel_control.model.forward_one_step(torch.cat((u_t, torch.zeros((1, y_target.size(1)), device=device)), dim=1), h_t)
    # h_t_seq.append(h_t.detach().cpu().numpy()[0])

    for t_step in range(sim_steps):
        target_index = np.searchsorted(step_break, t_step, side='right') - 1

        # feedforward to the simulation model
        u_t_sim = u_t.clone().detach()
        if t_step > 0 :
            last_y_pred_sim_t = y_pred_sim_t.clone()
        y_pred_sim_t, h_t_sim = crtlModel_sim.model.forward_one_step(torch.cat((u_t_sim, torch.zeros_like(y_target[target_index:target_index+1])), dim=1), h_t_sim)
        output_y_seq.append(y_pred_sim_t)
        h_t_sim_seq.append(h_t_sim.detach().cpu().numpy()[0])

        # feedforward to the control model, to obtrain the Jacobian
        input_u_seq.append(u_t.detach().cpu().numpy()[0])
        u_t = u_t.clone().detach().requires_grad_(True)  # ensure u_t is differentiable
        # y_pred_control_t, h_t = crtlModel_control.model.forward_one_step(torch.cat((u_t, torch.zeros_like(y_target[target_index:target_index+1])), dim=1), h_t)
        if t_step==0:
            y_pred_control_t, h_t = crtlModel_control.model.forward_one_step(torch.cat((u_t, torch.zeros((1, y_target.size(1)), device=device)), dim=1), h_t)
        else:
            error_y_pred = last_y_pred_sim_t - y_pred_control_t
            error_y_pred_seq.append(error_y_pred.detach().cpu().numpy()[0])
            y_pred_control_t, h_t = crtlModel_control.model.forward_one_step(torch.cat((u_t, error_y_pred), dim=1), h_t)
        h_t_seq.append(h_t.detach().cpu().numpy()[0])
        # Compute Jacobian dy/du
        jacobian = []
        for i in range(y_pred_control_t.size(1)):
            grad = torch.autograd.grad(y_pred_control_t[0, i], u_t, retain_graph=True)[0]
            jacobian.append(grad[0].detach())
        J = torch.stack(jacobian, dim=0)  # Shape: (output_dim, input_dim)

        # Compute the desired y
        # delta_y_desired = (y_target[target_index] - y_pred_sim_t).detach().T
        delta_y_desired = (y_target[target_index] - y_pred_control_t).detach().T

        # Solve mixed input correction
        # delta_u = mixed_input_correction(
        #     J, delta_y_desired, u_t[:, 0], u_t[:, 1],
        #     crtlModel_control.model, h_t, alpha, y_target[target_index],
        #     disc_search_radius=feedrate_delta,
        #     lambda_smooth=lambda_smooth, lambda_disc=lambda_disc
        # )
        # u_t = u_t + alpha * delta_u  # Update u_t with the correction
        # u_t[:,1] = torch.round(u_t[:, 1] / feedrate_delta) * feedrate_delta  # Discretize the feedrate input

        u_cont_new, u_disc_new = crtlModel_control.mixed_input_correction(
            J, delta_y_desired, u_t[:, 0], u_t[:, 1],
            h_t, alpha, y_target[target_index],
            lambda_smooth=lambda_smooth, lambda_disc=lambda_disc
        )
        u_t_cont_new = u_t[:, 0] + alpha * (u_cont_new - u_t[:, 0])
        u_t = torch.stack([u_t_cont_new, u_disc_new], dim=1)
    
    # # plot h_t_sim_seq to visualize the hidden state evolution
    # h_t_sim_seq = np.array(h_t_sim_seq)
    # plt.figure(figsize=(12, 6))
    # plt.plot(h_t_sim_seq, '-o')
    # plt.show()
    # # plot error_y_pred_seq to visualize the error evolution
    # error_y_pred_seq = np.array(error_y_pred_seq)
    # plt.figure(figsize=(12, 6))
    # plt.plot(error_y_pred_seq, '-o')
    # plt.show()

    # plot the results output_y_seq and y target vs time
    output_y_seq = torch.stack(output_y_seq, dim=0).detach().cpu().numpy()
    input_u_seq = np.array(input_u_seq)
    input_u_seq = crtlModel_control.denormalize_input(input_u_seq)
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