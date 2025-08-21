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

np.random.seed(42) # for reproducibility
torch.manual_seed(42) # for reproducibility

mm2inch = 1/25.4
inch2mm = 25.4

class controlLogLogModel():
    def __init__(self,model_dir,lambda_fac=0.99,cov_dh_init=10,cov_dw_init=10):
        self.model_dir = 'weld_Seq_models/'+model_dir

        self.theta_dh = np.loadtxt(self.model_dir + '/theta_param_dh.csv', delimiter=',')
        self.theta_dw = np.loadtxt(self.model_dir + '/theta_param_dw.csv', delimiter=',')
        self.theta_dh_origin = deepcopy(self.theta_dh)
        self.theta_dw_origin = deepcopy(self.theta_dw)

        self.lambda_fac = lambda_fac
        self.P_dh = np.eye(3)*cov_dh_init
        self.P_dw = np.eye(3)*cov_dw_init
        self.theta_dh_history = []
        self.theta_dw_history = []

    def get_control_loglog(self, dh, dw):

        dh = np.max([0.03, dh]) # prevent from log invalid
        dw = np.max([0.03, dw]) # prevent from log invalid

        log_v_om = np.linalg.pinv(np.vstack((self.theta_dh[:2], self.theta_dw[:2])))@(np.log([dh,dw])-np.array([self.theta_dh[2],self.theta_dw[2]]))

        torch_v = np.exp(log_v_om[0])
        torch_feedrate = np.exp(log_v_om[1])
        torch_feedrate = torch_feedrate * mm2inch * 60 # mm/s to inch/min
        origin_vpd_ratio = torch_v/torch_feedrate
        torch_feedrate = np.clip(torch_feedrate, 50, 250)
        torch_feedrate = round(torch_feedrate/10)*10 # round to nearest 10
        torch_v = torch_feedrate*origin_vpd_ratio

        return torch_v, torch_feedrate

    def get_pred_loglog(self, torch_v, feedrate):

        torch_v = np.max([0.03, torch_v]) # prevent from log invalid
        feedrate = np.max([10, feedrate]) # prevent from log invalid

        torch_feedrate = feedrate * inch2mm / 60 # inch/min to mm/s
        torch_v_log = np.log(torch_v)
        torch_feedrate_log = np.log(torch_feedrate)

        dh_pred = np.exp(self.theta_dh[0]*torch_v_log + self.theta_dh[1]*torch_feedrate_log + self.theta_dh[2])
        dw_pred = np.exp(self.theta_dw[0]*torch_v_log + self.theta_dw[1]*torch_feedrate_log + self.theta_dw[2])

        return dh_pred, dw_pred

    def rls_update(self, profile_height, last_profile_height, profile_width, control_inputs):

        # ignore the edge of the walls
        start_end_location = 47.5
        control_inputs = control_inputs[control_inputs[:,1]>=-start_end_location]
        control_inputs = control_inputs[control_inputs[:,1]<=start_end_location]
        # interp to get the dh width
        last_measured_height = np.interp(control_inputs[:,1], last_profile_height[:,0], last_profile_height[:,1])
        this_measured_height = np.interp(control_inputs[:,1], profile_height[:,0], profile_height[:,1])
        measured_dh = this_measured_height - last_measured_height
        measured_width = np.interp(control_inputs[:,1], profile_width[:,0], profile_width[:,1])

        cmd_updated_id = np.where(control_inputs[:,-1]!=0)[0]

        if cmd_updated_id.size>0:
            print("Get multiple new data points. RLS update.")
            ##### sample data
            measured_dh_sample = self._get_sum_profile(measured_dh, cmd_updated_id)
            measured_width_sample = self._get_sum_profile(measured_width, cmd_updated_id)
            control_torch_v_sample = self._get_sum_profile(control_inputs[:,2], cmd_updated_id)
            control_feedrate_sample = self._get_sum_profile(control_inputs[:,3], cmd_updated_id)
            control_feedrate_sample = control_feedrate_sample*inch2mm/60 # from ipm to mm/sec

            ##### prepare data
            measured_width_sample = measured_width_sample[measured_dh_sample>0]
            control_torch_v_sample = control_torch_v_sample[measured_dh_sample>0]
            control_feedrate_sample = control_feedrate_sample[measured_dh_sample>0]
            measured_dh_sample = measured_dh_sample[measured_dh_sample>0]
            measured_dh_sample = measured_dh_sample[measured_width_sample>0]
            control_torch_v_sample = control_torch_v_sample[measured_width_sample>0]
            control_feedrate_sample = control_feedrate_sample[measured_width_sample>0]
            measured_width_sample = measured_width_sample[measured_width_sample>0]

            # control input matrix
            X_new_input = np.vstack((np.log(control_torch_v_sample), np.log(control_feedrate_sample), np.ones_like(control_torch_v_sample))).T
            # measure output vectors
            measured_dh_sample_log = np.log(measured_dh_sample)
            measured_width_sample_log = np.log(measured_width_sample)

            ##### update the parameters using recursive least squares
            self.theta_dh_history.append(deepcopy(self.theta_dh))
            self.theta_dw_history.append(deepcopy(self.theta_dw))
            # update theta dh
            K_gain_dh = self.P_dh@X_new_input.T@np.linalg.inv(self.lambda_fac*np.eye(X_new_input.shape[0])+X_new_input@self.P_dh@X_new_input.T)
            self.theta_dh = self.theta_dh + K_gain_dh@(measured_dh_sample_log-X_new_input@self.theta_dh)
            self.P_dh = (self.P_dh-K_gain_dh@X_new_input@self.P_dh)/self.lambda_fac
            # update theta dw
            K_gain_dw = self.P_dw@X_new_input.T@np.linalg.inv(self.lambda_fac*np.eye(X_new_input.shape[0])+X_new_input@self.P_dw@X_new_input.T)
            self.theta_dw = self.theta_dw + K_gain_dw@(measured_width_sample_log-X_new_input@self.theta_dw)
            self.P_dw = (self.P_dw-K_gain_dw@X_new_input@self.P_dw)/self.lambda_fac

            print("Previous parameters:")
            print("theta_dh:", self.theta_dh_history[-1])
            print("theta_dw:", self.theta_dw_history[-1])
            print("Updated parameters:")
            print("theta_dh:", self.theta_dh)
            print("theta_dw:", self.theta_dw)

    def _get_sum_profile(self,profile,sample_id):

        profile_sum = np.concatenate(([0.0],np.cumsum(profile, dtype=np.float64)))
        starts, ends = sample_id[:-1], sample_id[1:]
        sums = profile_sum[ends] - profile_sum[starts]
        counts = ends - starts
        means = sums / counts
        return means

class controlModel():
    def __init__(self, model_dir, minmax_v_file='weld_Seq_models/test_cmd_v_feedrate.csv', minmax_feedrate_file='weld_Seq_models/train_cmd_v_feedrate.csv',\
                  device='cpu'):

        # for neural network input normalization
        all_inputs = np.loadtxt(minmax_v_file, delimiter=',',skiprows=1)
        all_inputs = np.vstack((all_inputs, np.loadtxt(minmax_feedrate_file, delimiter=',',skiprows=1)))
        self.v_max = np.max(all_inputs[:, 0])
        self.v_min = np.min(all_inputs[:, 0])
        self.feedrate_max = np.max(all_inputs[:, 1])
        self.feedrate_min = np.min(all_inputs[:, 1])
        print(f"v_min: {self.v_min}, v_max: {self.v_max}, feedrate_min: {self.feedrate_min}, feedrate_max: {self.feedrate_max}")
        self.feedrate_delta = 10 / (self.feedrate_max-self.feedrate_min) # 10 is discretized feedrate value

        self.model_dir = model_dir
        self.device = device
        self.model, self.model_params = self.load_model()

        self.initialize_state()
    
    def initialize_state(self):
        self.h_t = torch.zeros((1, self.model_params['model_hidden_size']), device=self.device)

    def load_model(self):
        model_dir = 'weld_Seq_models/'+self.model_dir+'/'
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
            model = modelClass(input_size=model_input_size, hidden_size=model_hidden_size, output_size=model_output_size, num_layers=num_layers, device=self.device).to(self.device)
        else:
            model = modelClass(input_size=model_input_size, hidden_size=model_hidden_size, output_size=model_output_size, num_layers=num_layers, history_length=history_length, latency_steps=latency_steps, open_loop=open_loop, device=self.device).to(self.device)

        # load the model state
        model_state_dict = torch.load(model_dir + 'best_model.pth', map_location=self.device, weights_only=True)
        model.load_state_dict(model_state_dict)

        return model, training_params

    def get_model(self):
        return self.model, self.model_params
    
    def normalize_input(self,u):
        """
        Normalize the input based on the predefined min and max values.
        """
        u_in = deepcopy(u)
        u_in[:, 0] = (u_in[:, 0] - self.v_min) / (self.v_max - self.v_min)
        u_in[:, 1] = (u_in[:, 1] - self.feedrate_min) / (self.feedrate_max - self.feedrate_min)
        return u_in

    def denormalize_input(self,u_normed):
        """
        Denormalize the input based on the predefined min and max values.
        """
        u = np.zeros_like(u_normed)
        u[:, 0] = u_normed[:, 0] * (self.v_max - self.v_min) + self.v_min
        u[:, 1] = u_normed[:, 1] * (self.feedrate_max - self.feedrate_min) + self.feedrate_min
        return u

    def forward_one_step(self,torch_v,torch_feedrate,h_t=None,error_measure=None):

        u_t = torch.tensor(self.normalize_input(np.array([[torch_v, torch_feedrate]],dtype=np.float64)), dtype=torch.float32, device=self.device)
        u_t = u_t.clone().detach().requires_grad_(True)  # ensure u_t is differentiable

        if h_t is None:
            h_t = self.h_t
        if error_measure is None:
            y_pred_t, h_t = self.model.forward_one_step(torch.cat((u_t, torch.zeros((1, 2), device=self.device)), dim=1), h_t)
        else:
            y_pred_t, h_t = self.model.forward_one_step(torch.cat((u_t, error_measure), dim=1), h_t)
        self.h_t = h_t

        return y_pred_t, h_t, u_t

    def forward_one_step_get_opt_u(self, torch_v, torch_feedrate, dh_target, dw_target, alpha, h_t=None, error_measure=None, lambda_smooth=1e-2, lambda_disc=1e-2):

        y_pred_t, h_t, u_t = self.forward_one_step(torch_v, torch_feedrate, h_t=h_t, error_measure=error_measure)

        # Compute Jacobian dy/du
        jacobian = []
        for i in range(y_pred_t.size(1)):
            grad = torch.autograd.grad(y_pred_t[0, i], u_t, retain_graph=True)[0]
            jacobian.append(grad[0].detach())
        J = torch.stack(jacobian, dim=0)  # Shape: (output_dim, input_dim)

        # Compute the desired y
        y_target = torch.tensor([[dh_target, dw_target]], dtype=torch.float32, device=self.device)
        delta_y_desired = (y_target - y_pred_t).detach().T

        # Solve mixed input correction
        # delta_u = mixed_input_correction(
        #     J, delta_y_desired, u_t[:, 0], u_t[:, 1],
        #     crtlModel_control.model, h_t, alpha, y_target[target_index],
        #     disc_search_radius=feedrate_delta,
        #     lambda_smooth=lambda_smooth, lambda_disc=lambda_disc
        # )
        # u_t = u_t + alpha * delta_u  # Update u_t with the correction
        # u_t[:,1] = torch.round(u_t[:, 1] / feedrate_delta) * feedrate_delta  # Discretize the feedrate input

        u_cont_new, u_disc_new = self.mixed_input_correction(
            J, delta_y_desired, u_t[:, 0], u_t[:, 1],
            h_t, alpha, y_target,
            lambda_smooth=lambda_smooth, lambda_disc=lambda_disc
        )
        u_t_cont_new = u_t[:, 0] + alpha * (u_cont_new - u_t[:, 0])
        u_t = torch.stack([u_t_cont_new, u_disc_new], dim=1)
        u_t_denorm = self.denormalize_input(u_t.detach().cpu().numpy())[0]
        dh_pred, dw_pred = y_pred_t.detach().cpu().numpy()[0]
        return u_t_denorm[0], u_t_denorm[1], dh_pred, dw_pred

    def mixed_input_correction(self,J:torch.tensor, delta_y:torch.tensor, u_cont_prev:torch.tensor, u_disc_prev:torch.tensor,
                            h_t, alpha, y_desired,disc_search_radius=None, lambda_smooth=1e-2, lambda_disc=1e-2):
        """
        J: Jacobian matrix, shape (output_dim, input_dim)
        delta_y: Desired output change, shape (output_dim, 1)
        u_cont_prev: previous continuous input (torch speed) tensor of shape (cont_dim,)
        u_disc_prev: previous discrete input (wire feed rate) scalar

        Returns: Updated (u_cont_new, u_disc_new)
        """
        if disc_search_radius is None:
            disc_search_radius = self.feedrate_delta

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
            if u_disc_candidate < 0 or u_disc_candidate > 1:
                # higher or lower than maximum values
                continue

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
