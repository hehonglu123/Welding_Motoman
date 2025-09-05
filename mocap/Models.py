import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from calib_analytic_grad import *
from robotics_utils import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Custom Weighted MSE Loss for element-wise weighting
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, input, target, weights):
        # Calculate the squared difference
        diff = input - target
        squared_diff = diff ** 2
        # Multiply each element by its corresponding weight across the output dimensions
        weighted_squared_diff = squared_diff * weights
        # Return the mean of the weighted squared differences
        loss = weighted_squared_diff.mean()
        return loss

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[20,20]):
        super(NeuralNetwork, self).__init__()

        self.hiddenLayers = nn.ModuleList()
        self.relus = nn.ModuleList()
        for k in range(len(hidden_sizes)):
            if k == 0:
                self.hiddenLayers.append(nn.Linear(input_size, hidden_sizes[k]))
            else:
                self.hiddenLayers.append(nn.Linear(hidden_sizes[k-1], hidden_sizes[k]))
            self.relus.append(nn.ReLU())
        self.output = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        for k in range(len(self.hiddenLayers)):
            x = self.hiddenLayers[k](x)
            x = self.relus[k](x)
        x = self.output(x)
        return x

    def forward_features(self, x):
        for k in range(len(self.hiddenLayers)):
            x = self.hiddenLayers[k](x)
            x = self.relus[k](x)
        return x

class NNVariationalEncoder(nn.Module):
    def __init__(self, data_size, latent_size, hidden_sizes=[20,20], mu=0, sigma=1):
        super(NNVariationalEncoder, self).__init__()

        self.hiddenLayers = nn.ModuleList()
        self.relus = nn.ModuleList()
        for k in range(len(hidden_sizes)):
            if k == 0:
                self.hiddenLayers.append(nn.Linear(data_size, hidden_sizes[k]))
            else:
                self.hiddenLayers.append(nn.Linear(hidden_sizes[k-1], hidden_sizes[k]))
            self.relus.append(nn.ReLU())
        self.output_mu = nn.Linear(hidden_sizes[-1], latent_size)
        self.output_sigma = nn.Linear(hidden_sizes[-1], latent_size)

        self.latent_mu = torch.tensor(mu)
        self.latent_sigma = torch.tensor(sigma)
        self.N = torch.distributions.Normal(mu, sigma)
        self.kl = 0

    def forward(self, x):
        for k in range(len(self.hiddenLayers)):
            x = self.hiddenLayers[k](x)
            x = self.relus[k](x)
        mu = self.output_mu(x)
        sigma = torch.exp(self.output_sigma(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = ((sigma**2 + (mu-self.latent_mu)**2)/(2*self.latent_sigma**2) + torch.log(self.latent_sigma/sigma) - 1/2).sum()
        return z

class VariationalAutoEncoder(nn.Module):
    def __init__(self, data_size, latent_size, hidden_sizes=[20,20], mu=0, sigma=1):
        super(VariationalAutoEncoder, self).__init__()

        self.encoder = NNVariationalEncoder(data_size, latent_size, hidden_sizes, mu, sigma)
        self.decoder = NeuralNetwork(latent_size, data_size, hidden_sizes[::-1])
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
class AutoEncoder(nn.Module):
    def __init__(self, data_size, latent_size, hidden_sizes=[20,20]):
        super(AutoEncoder, self).__init__()

        self.encoder = NeuralNetwork(data_size, latent_size, hidden_sizes)
        self.decoder = NeuralNetwork(latent_size, data_size, hidden_sizes[::-1])
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
class FourierNetwork(nn.Module):

    def __init__(self, input_size, output_size):
        super(FourierNetwork, self).__init__()

        self.output = nn.Linear(12, output_size)
    
    def forward(self,x):
        if len(x.shape) == 2:
            sum_input = torch.sum(x,dim=1,keepdim=True)
            x = torch.cat((torch.sin(x),torch.cos(x),torch.sin(sum_input),torch.cos(sum_input),\
                               torch.sin(2*x),torch.cos(2*x),torch.sin(2*sum_input),torch.cos(2*sum_input)),dim=1)
        else:
            sum_input = torch.Tensor([torch.sum(x)])
            x = torch.cat((torch.sin(x),torch.cos(x),torch.sin(sum_input),torch.cos(sum_input),\
                                torch.sin(2*x),torch.cos(2*x),torch.sin(2*sum_input),torch.cos(2*sum_input)))
        x = self.output(x)
        return x
    
    def forward_features(self,x):
        if len(x.shape) == 2:
            sum_input = torch.sum(x,dim=1,keepdim=True)
            x = torch.cat((torch.sin(x),torch.cos(x),torch.sin(sum_input),torch.cos(sum_input),\
                               torch.sin(2*x),torch.cos(2*x),torch.sin(2*sum_input),torch.cos(2*sum_input)),dim=1)
        else:
            sum_input = torch.Tensor([torch.sum(x)])
            x = torch.cat((torch.sin(x),torch.cos(x),torch.sin(sum_input),torch.cos(sum_input),\
                                torch.sin(2*x),torch.cos(2*x),torch.sin(2*sum_input),torch.cos(2*sum_input)))
        return x

class NeuralFourierNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[20,20]):
        super(NeuralFourierNetwork, self).__init__()

        self.hiddenLayers = nn.ModuleList()
        self.relus = nn.ModuleList()
        for k in range(len(hidden_sizes)):
            if k == 0:
                self.hiddenLayers.append(nn.Linear(input_size, hidden_sizes[k]))
            else:
                self.hiddenLayers.append(nn.Linear(hidden_sizes[k-1], hidden_sizes[k]))
            self.relus.append(nn.ReLU())
        self.output = nn.Linear(hidden_sizes[-1]+12, output_size)
        # define a fourier layer
        

    def forward(self, x):

        if len(x.shape) == 2:
            sum_input = torch.sum(x,dim=1,keepdim=True)
            fourier_x = torch.cat((torch.sin(x),torch.cos(x),torch.sin(sum_input),torch.cos(sum_input),\
                               torch.sin(2*x),torch.cos(2*x),torch.sin(2*sum_input),torch.cos(2*sum_input)),dim=1)
        else:
            sum_input = torch.Tensor([torch.sum(x)])
            fourier_x = torch.cat((torch.sin(x),torch.cos(x),torch.sin(sum_input),torch.cos(sum_input),\
                                torch.sin(2*x),torch.cos(2*x),torch.sin(2*sum_input),torch.cos(2*sum_input)))
        for k in range(len(self.hiddenLayers)):
            x = self.hiddenLayers[k](x)
            x = self.relus[k](x)
        if len(x.shape) == 2:
            x = torch.cat((x,fourier_x),dim=1)
        else:
            x = torch.cat((x,fourier_x))
        x = self.output(x)
        return x
    
# Custom transformation error with manually specified gradient, the analytical gradient using autograd.Function
class TransformationLossFunction(Function):
    @staticmethod
    def forward(ctx, predict_PH, target, joint_angles, robot, param_nominal, weight_pos=1, weight_ori=1):

        p_error_all = []
        ori_error_all = []
        for i,(q,ph,T) in enumerate(zip(joint_angles,predict_PH,target)):
            robot = get_PH_from_param(ph.detach().numpy()+param_nominal,robot,unit='radians')
            T_pred = robot.fwd(q)
            p_error = T_pred.p - T.p
            # omega_d= s_err_func(T_pred.R@T.R.T)
            k,theta = R2rot(T_pred.R@T.R.T)
            omega_d= k*theta
            p_error_all.append(p_error)
            ori_error_all.append(omega_d)
        loss = torch.tensor(np.mean(weight_pos*np.linalg.norm(p_error_all,axis=1)+weight_ori*np.linalg.norm(ori_error_all,axis=1)))

        # save additional arguments for backward
        ctx.save_for_backward(predict_PH)
        ctx.additional_args = p_error_all, ori_error_all, joint_angles, robot, param_nominal, weight_pos, weight_ori

        return loss, p_error_all, ori_error_all

    @staticmethod
    def backward(ctx, grad_output, dum_a, dum_b):

        predict_PH, = ctx.saved_tensors
        p_error_all, ori_error_all, joint_angles, robot, param_nominal, weight_pos, weight_ori = ctx.additional_args

        grad = []
        N = len(predict_PH)
        for i,(q,ph,p_error,ori_error) in enumerate(zip(joint_angles,predict_PH,p_error_all,ori_error_all)):
            J_ana_part = jacobian_param(ph.detach().numpy()+param_nominal,robot,q)
            mu = np.append(ori_error*weight_ori/N,p_error*weight_pos/N)
            grad.append(torch.tensor(np.dot(mu,J_ana_part)))
        
        return torch.stack(grad), None, None, None, None, None, None

# Custom loss class that inherits from nn.Module
class TransformationLoss(nn.Module):
    def __init__(self):
        super(TransformationLoss, self).__init__()

    def forward(self, predict_PH, target, joint_angles, robot, param_nominal, weight_pos=1, weight_ori=1):
        # Use the custom autograd function for the forward pass
        loss, p_error_all, ori_error_all = TransformationLossFunction.apply(predict_PH, target, joint_angles, robot, param_nominal, weight_pos, weight_ori)
        return loss, p_error_all, ori_error_all

# Neural Network with Tanh activation function for ARMA
class ARMANeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=[20,20], num_layers=1, history_length=1, open_loop=False, device='cpu'):
        super(ARMANeuralNetwork, self).__init__()
        self.history_length = history_length
        self.output_size = output_size
        self.open_loop = open_loop

        self.hiddenLayers = nn.ModuleList()
        self.tanh = nn.ModuleList()
        for k in range(len(hidden_size)):
            if k == 0:
                self.hiddenLayers.append(nn.Linear(input_size, hidden_size[k]))
            else:
                self.hiddenLayers.append(nn.Linear(hidden_size[k-1], hidden_size[k]))
            self.tanh.append(nn.Tanh())
        self.output = nn.Linear(hidden_size[-1], output_size)

    def forward(self, x_true, u):
        batch_size, seq_len, _ = x_true.size()
        predictions_input = torch.zeros((batch_size, self.history_length, self.output_size), device=x_true.device)
        predictions = []
        
        for t in range(self.history_length,seq_len):
            u_t = torch.flatten(u[:, t-self.history_length+1:t, :], start_dim=1)  # (batch, history_length * input_size)
            u_t = torch.cat([u[:, t, :], u_t], dim=1)
            this_pred = torch.flatten(predictions_input, start_dim=1)  # (batch, history_length * output_size)
            this_error = torch.flatten(x_true[:, t-self.history_length:t, :]-predictions_input, start_dim=1)  # (batch, history_length * output_size)
            # y_t = torch.flatten(x_true[:, t-self.history_length:t, :], start_dim=1)
            # Concatenate error and inputs
            if not self.open_loop:
                x = torch.cat([this_pred, u_t, this_error], dim=1) # (batch, history_length * output_size + current_input + history_length * input_size + error)
            else:
                x = torch.cat([this_pred, u_t], dim=1) # (batch, history_length * output_size + current_input + history_length * input_size)
                
            for k in range(len(self.hiddenLayers)):
                x = self.hiddenLayers[k](x)
                x = self.tanh[k](x)
            x = self.output(x)
            predictions.append(x)
            predictions_input = torch.cat([predictions_input[:, 1:, :], x.unsqueeze(1)], dim=1)  # Shift the predictions input
        predictions = torch.stack(predictions, dim=1)  # (batch, seq_len - history_length, output_size)
        return predictions

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, device='cpu'):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, x_true, u):
        h0 = torch.zeros(self.num_layers, u.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, u.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(u, (h0, c0))
        out = self.fc(out)  # Get the last time step's output
        return out

# LSTM model for autoregression
class LSTMAutoRegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, history_length=0, open_loop=False, device='cpu'):
        super(LSTMAutoRegressionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.history_length = history_length
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, x_true, u):
        batch_size, seq_len, _ = x_true.size()

        # Initial hidden states
        h_t = torch.zeros(batch_size, self.hidden_size, device=self.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=self.device)
        # Initial error is zero
        error = torch.zeros_like(x_true[:, 0, :])  
        predictions = []

        for t in range(self.history_length,seq_len):
            u_t = torch.flatten(u[:, t-self.history_length+1:t, :],start_dim=1)  # (batch, history_length * input_size)
            u_t = torch.cat([u[:, t, :], u_t], dim=1)  # (batch, current_input + history_length * input_size)
            y_t = torch.flatten(x_true[:, t-self.history_length:t, :], start_dim=1)  # (batch, history_length * output_size)
            # Concatenate error and inputs
            input_t = torch.cat([y_t,u_t,error], dim=1)  # (batch, history_length * output_size + current_input + history_length * input_size + error)

            h_t, c_t = self.lstm_cell(input_t, (h_t, c_t))
            y_pred = self.fc(h_t)
            predictions.append(y_pred)

            error = x_true[:, t, :] - y_pred

        predictions = torch.stack(predictions, dim=1)
        return predictions

# RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, device='cpu'):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, x_true, u):
        h0 = torch.zeros(self.num_layers, u.size(0), self.hidden_size).to(self.device)
        out, _ = self.rnn(u, h0)
        out = self.fc(out)  # Get the last time step's output
        return out

class KalmanNet(nn.Module):
    def __init__(self, state_dim, observation_dim, Qgru_input_dim, Pgru_input_dim, Sgru_input_dim):
        super().__init__()
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.Qfc = nn.Linear(state_dim, Qgru_input_dim)
        self.Qgru = nn.GRUCell(Qgru_input_dim, state_dim**2)
        self.Pfc = nn.Linear(state_dim, Pgru_input_dim)
        self.Pgru = nn.GRUCell(Pgru_input_dim+state_dim**2, state_dim**2)
        self.fc_P2S = nn.Linear(state_dim**2, Sgru_input_dim)
        self.Sfc = nn.Linear(observation_dim*2, Sgru_input_dim)
        self.Sgru = nn.GRUCell(Sgru_input_dim*2, observation_dim**2)
        self.fc2K = nn.Linear(observation_dim**2+state_dim**2, state_dim*observation_dim)
        self.fc2P1 = nn.Linear(state_dim*observation_dim+observation_dim**2,state_dim*observation_dim)
        self.fc2P2 = nn.Linear(state_dim*observation_dim+state_dim**2,state_dim*state_dim)

    def forward(self, obs_diff, inno_diff, evo_diff, upd_diff, Q_prev, P_prev, S_prev):
        Q_new = self.Qgru(torch.tanh(self.Qfc(upd_diff)), Q_prev)
        P_new = self.Pgru(torch.cat([torch.tanh(self.Pfc(evo_diff)), Q_new], dim=1), P_prev)
        all_obs_input = torch.cat([obs_diff, inno_diff], dim=1)  # Concatenate observation and innovation differences
        S_new = self.Sgru(torch.cat([torch.tanh(self.Sfc(all_obs_input)), torch.tanh(self.fc_P2S(P_new))], dim=1), S_prev)
        K_new = torch.tanh(self.fc2K(torch.cat([S_new, Q_new], dim=1)))
        P_new = torch.tanh(self.fc2P1(torch.cat([K_new, S_new], dim=1)))
        P_new = self.fc2P2(torch.cat([P_new, Q_new], dim=1))
        return K_new, P_new, S_new, Q_new

class DeepTransitionRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, depth=2):
        super().__init__()
        self.depth = depth
        self.input_size = input_size
        self.hidden_size = hidden_size

        # First layer takes input + hidden
        self.in_layer = nn.Linear(input_size + hidden_size, hidden_size)

        # Additional transition layers (hidden -> hidden)
        self.transition_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(depth - 1)
        ])

    def forward(self, x_t, h_prev):
        h = torch.cat([x_t, h_prev], dim=1)
        h = torch.tanh(self.in_layer(h))
        for layer in self.transition_layers:
            h = torch.tanh(layer(h))
        return h

# RNN Model for autoregression
class RNNAutoRegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, history_length=0, latency_steps=0, open_loop=False, device='cpu'):
        super(RNNAutoRegressionModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.history_length = history_length
        self.latency_steps = latency_steps
        self.open_loop = open_loop
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        if num_layers > 1:
            self.rnn_cell = DeepTransitionRNNCell(input_size, hidden_size, depth=num_layers)
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = device
    
    def forward(self, x_true, u):
        batch_size, seq_len, _ = x_true.size()

        # Initial hidden states
        h_t = torch.zeros(batch_size, self.hidden_size, device=self.device)
        # Initial error is zero
        error = torch.zeros_like(x_true[:, 0, :],device=self.device)
        predictions = []

        for t in range(self.history_length, seq_len):
            if not self.open_loop:
                u_t = torch.flatten(u[:, t-self.history_length+1:t, :], start_dim=1)  # (batch, history_length * input_size)
                u_t = torch.cat([u[:, t, :], u_t], dim=1)
                if t >= self.latency_steps:
                    y_t = torch.flatten(x_true[:, t-self.history_length-self.latency_steps:t-self.latency_steps, :], start_dim=1)  # (batch, history_length * output_size)
                else:
                    y_t = torch.zeros_like(torch.flatten(x_true[:, t-self.history_length:t, :], start_dim=1), device=self.device)  # (batch, history_length * output_size)
                # Concatenate error and inputs
                input_t = torch.cat([y_t, u_t, error], dim=1)  # (batch, history_length * output_size + current_input + history_length * input_size + error)
            else:
                input_t = u[:, t, :]  # (batch, input_size)

            h_t = self.rnn_cell(input_t, h_t)
            y_pred = self.fc(h_t)
            predictions.append(y_pred)

            if t>= self.latency_steps:
                error = x_true[:, t-self.latency_steps, :] - predictions[-self.latency_steps-1]  # Use the prediction from latency steps ago
            else:
                error = torch.zeros_like(y_pred, device=self.device)

        predictions = torch.stack(predictions, dim=1)
        return predictions

    def forward_linear_mat(self, x_true, u):
        batch_size, seq_len, _ = x_true.size()

        with torch.no_grad():
            fc_hh = nn.Linear(self.hidden_size, self.hidden_size).to(self.device)
            fc_ih = nn.Linear(self.input_size, self.hidden_size).to(self.device)
            fc_hh.weight.copy_(self.rnn_cell.weight_hh)
            fc_hh.bias.copy_(self.rnn_cell.bias_hh)
            fc_ih.weight.copy_(self.rnn_cell.weight_ih)
            fc_ih.bias.copy_(self.rnn_cell.bias_ih)

        # Initial hidden states
        h_t = torch.zeros(batch_size, self.hidden_size, device=self.device)
        # Initial error is zero
        error = torch.zeros_like(x_true[:, 0, :],device=self.device)
        predictions = []
        hidden_linear_zt = []

        for t in range(self.history_length, seq_len):
            if not self.open_loop:
                u_t = torch.flatten(u[:, t-self.history_length+1:t, :], start_dim=1)  # (batch, history_length * input_size)
                u_t = torch.cat([u[:, t, :], u_t], dim=1)
                y_t = torch.flatten(x_true[:, t-self.history_length:t, :], start_dim=1)  # (batch, history_length * output_size)
                # Concatenate error and inputs
                input_t = torch.cat([y_t, u_t, error], dim=1)  # (batch, history_length * output_size + current_input + history_length * input_size + error)
            else:
                input_t = u[:, t, :]  # (batch, input_size)

            z_t = fc_ih(input_t) + fc_hh(h_t)
            h_t = torch.tanh(z_t)  # Apply tanh activation
            # h_t = self.rnn_cell(input_t, h_t)
            y_pred = self.fc(h_t)
            hidden_linear_zt.append(z_t)
            predictions.append(y_pred)

            error = x_true[:, t, :] - y_pred

        hidden_linear_zt = torch.stack(hidden_linear_zt, dim=1)  # (batch, seq_len - history_length, hidden_size)
        predictions = torch.stack(predictions, dim=1)
        return predictions, hidden_linear_zt
    
    def forward_half_obs(self, x_true, u, h_t=None):
        batch_size, x_true_len, _ = x_true.size()
        _, u_len, _ = u.size()

        with torch.no_grad():
            fc_hh = nn.Linear(self.hidden_size, self.hidden_size).to(self.device)
            fc_ih = nn.Linear(2, self.hidden_size).to(self.device)
            fc_hh.weight.copy_(self.rnn_cell.weight_hh)
            fc_hh.bias.copy_(self.rnn_cell.bias_hh)
            fc_ih.weight.copy_(self.rnn_cell.weight_ih[:, :2])  # Use only the first two input features
            fc_ih.bias.copy_(self.rnn_cell.bias_ih)

        # Initial hidden states
        if h_t is None:
            # If h_t is not provided, initialize it
            h_t = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initial error is zero
        if x_true_len > 0:
            error = torch.zeros_like(x_true[:, 0, :],device=self.device)
        predictions = []

        for t in range(self.history_length, u_len):
            if not self.open_loop:
                u_t = torch.flatten(u[:, t-self.history_length+1:t, :], start_dim=1)  # (batch, history_length * input_size)
                u_t = torch.cat([u[:, t, :], u_t], dim=1)
                if t < x_true_len:
                    y_t = torch.flatten(x_true[:, t-self.history_length:t, :], start_dim=1)  # (batch, history_length * output_size)
                    # Concatenate error and inputs
                    input_t = torch.cat([y_t, u_t, error], dim=1)  # (batch, history_length * output_size + current_input + history_length * input_size + error)
            else:
                input_t = u[:, t, :]  # (batch, input_size)

            if t < x_true_len or self.open_loop:
                h_t = self.rnn_cell(input_t, h_t)
            else:
                h_t = torch.tanh(fc_ih(u_t) + fc_hh(h_t))  # Apply tanh activation
            y_pred = self.fc(h_t)
            predictions.append(y_pred)

            if t < x_true_len:
                error = x_true[:, t, :] - y_pred

        predictions = torch.stack(predictions, dim=1)
        return predictions, h_t
    
    def forward_multi_steps(self,x_true,u,multi_steps=1):
        assert multi_steps > 0, "multi_steps must be greater than 0"

        batch_size, x_true_len, _ = x_true.size()
        _, u_len, _ = u.size()
        
        # Initial hidden states
        h_t = torch.zeros(batch_size, self.hidden_size, device=self.device)
        # Initial error is zero
        if x_true_len > 0:
            error = torch.zeros_like(x_true[:, 0, :],device=self.device)
        predictions = []
        predictions_one_step = []

        for t in range(self.history_length, u_len-multi_steps):
            if not self.open_loop:
                u_t = torch.flatten(u[:, t-self.history_length+1:t, :], start_dim=1)  # (batch, history_length * input_size)
                u_t = torch.cat([u[:, t, :], u_t], dim=1)
                if t >= self.latency_steps:
                    y_t = torch.flatten(x_true[:, t-self.history_length-self.latency_steps:t-self.latency_steps, :], start_dim=1)  # (batch, history_length * output_size)
                else:
                    y_t = torch.zeros_like(torch.flatten(x_true[:, t-self.history_length:t, :], start_dim=1), device=self.device)  # (batch, history_length * output_size)
                # Concatenate error and inputs
                input_t = torch.cat([y_t, u_t, error], dim=1)  # (batch, history_length * output_size + current_input + history_length * input_size + error)
            else:
                input_t = u[:, t, :]  # (batch, input_size)

            h_t = self.rnn_cell(input_t, h_t)
            y_pred = self.fc(h_t)
            predictions_one_step.append(y_pred)

            if t>= self.latency_steps:
                error = x_true[:, t-self.latency_steps, :] - predictions_one_step[-self.latency_steps-1]  # Use the prediction from latency steps ago
            else:
                error = torch.zeros_like(y_pred, device=self.device)
            
            # Multi-step prediction
            h_t_next = h_t.clone()
            for step in range(1,multi_steps+1):
                y_t_next = torch.zeros_like(torch.flatten(x_true[:, t-self.history_length:t, :], start_dim=1), device=self.device)
                error_next = torch.zeros_like(y_pred, device=self.device)
                u_t_next = torch.flatten(u[:, t-self.history_length+1+step:t+step, :], start_dim=1)  # (batch, history_length * input_size)
                u_t_next = torch.cat([u[:, t+step, :], u_t_next], dim=1)
                input_t_next = torch.cat([y_t_next, u_t_next, error_next], dim=1)  # (batch, output_size + current_input + input_size + error)
                h_t_next = self.rnn_cell(input_t_next, h_t_next)
                y_pred_next = self.fc(h_t_next)
            predictions.append(y_pred_next)

        predictions = torch.stack(predictions, dim=1)
        predictions_one_step = torch.stack(predictions_one_step, dim=1)
        return predictions, predictions_one_step

    def forward_one_step(self, u, h_t):

        h_t = self.rnn_cell(u, h_t)
        y_pred = self.fc(h_t)
        return y_pred, h_t

# GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, device='cpu'):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, x_true, u):
        h0 = torch.zeros(self.num_layers, u.size(0), self.hidden_size).to(self.device)
        out, _ = self.gru(u, h0)
        out = self.fc(out)  # Get the last time step's output
        return out

# GRU Model for autoregression
class GRUAutoRegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, history_length=0, open_loop=False, device='cpu'):
        super(GRUAutoRegressionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.history_length = history_length
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = device
    
    def forward(self, x_true, u):
        batch_size, seq_len, _ = x_true.size()

        # Initial hidden states
        h_t = torch.zeros(batch_size, self.hidden_size, device=self.device)
        # Initial error is zero
        error = torch.zeros_like(x_true[:, 0, :])  
        predictions = []

        for t in range(self.history_length, seq_len):
            u_t = torch.flatten(u[:, t-self.history_length+1:t, :], start_dim=1)  # (batch, history_length * input_size)
            u_t = torch.cat([u[:, t, :], u_t], dim=1)
            y_t = torch.flatten(x_true[:, t-self.history_length:t, :], start_dim=1)
            # Concatenate error and inputs
            input_t = torch.cat([y_t, u_t, error], dim=1) # (batch, history_length * output_size + current_input + history_length * input_size + error)

            h_t = self.gru_cell(input_t, h_t)
            y_pred = self.fc(h_t)
            predictions.append(y_pred)

            error = x_true[:, t, :] - y_pred
        
        predictions = torch.stack(predictions, dim=1)
        return predictions
    
class ThermalEncoder(nn.Module):
    """
    1D-CNN with strides/dilations -> GAP + GMP -> Linear -> LayerNorm
    Input per step: (B, neighbor_length) float
    Output per step: (B, E_T) where E_T=64 (default)
    """
    def __init__(self, emb_dim: int = 64):
        super().__init__()
        # Conv stack expands channels, reduces length
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),  # L: neighbor_length -> 200
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, dilation=2),  # 200 -> 100
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=4, dilation=4),  # 100 -> ~50
            nn.ReLU(),
        )
        # Final projection to embedding dim
        # We will concat GAP (96) and GMP (96) -> 192, then Linear to emb_dim
        self.proj = nn.Linear(192, emb_dim)
        self.ln = nn.LayerNorm(emb_dim)
        self.emb_dim = emb_dim
    
    def forward(self, thermal_raw: torch.Tensor) -> torch.Tensor:
        """
        thermal_raw: (B, neighbor_length) or (B, 1, neighbor_length)
        returns: (B, emb_dim)
        """
        if thermal_raw.dim() == 2:
            x = thermal_raw.unsqueeze(1)  # (B,1,neighbor_length)
        else:
            x = thermal_raw               # (B,1,neighbor_length)
        feat = self.conv(x)               # (B,96,L~50)

        # Global Average Pooling & Global Max Pooling along length
        gap = feat.mean(dim=-1)           # (B,96)
        gmp = feat.amax(dim=-1)           # (B,96)
        pooled = torch.cat([gap, gmp], dim=-1)  # (B,192)

        z = self.proj(pooled)             # (B,emb_dim)
        return self.ln(z)

class ThermalThinEncoder(nn.Module):
    def __init__(self, emb_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),             # 1→16
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2, dilation=2),# 16→32
            nn.ReLU(),
            nn.Conv1d(32, 48, kernel_size=5, stride=2, padding=4, dilation=4),# 32→48
            nn.ReLU(),
        )
        # GAP only (48) -> Linear -> emb_dim (32)
        self.proj = nn.Linear(48, emb_dim)
        self.ln = nn.LayerNorm(emb_dim)

    def forward(self, thermal_raw: torch.Tensor) -> torch.Tensor:
        x = thermal_raw.unsqueeze(1) if thermal_raw.dim() == 2 else thermal_raw
        feat = self.conv(x)             # (B,48,L~50)
        gap = feat.mean(dim=-1)         # (B,48)
        z = self.proj(gap)              # (B,emb_dim=32)
        return self.ln(z)

class ThermalNNEncoder(nn.Module):
    def __init__(self, emb_dim: int = 32):
        super().__init__()
        # self.mlp = nn.Sequential(
        #     nn.Linear(400, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, emb_dim),
        #     nn.ReLU(),
        # )
        self.mlp = nn.Sequential(
            nn.AvgPool1d(kernel_size=4, stride=4),  # downsample 400 -> 100
            nn.Linear(100, emb_dim),
            nn.ReLU(),
        )
        self.ln = nn.LayerNorm(emb_dim)
        self.emb_dim = emb_dim
    def forward(self, thermal_raw: torch.Tensor) -> torch.Tensor:
        """
        thermal_raw: (B, neighbor_length) float
        returns: (B, emb_dim)
        """
        if thermal_raw.dim() != 2:
            raise ValueError("ThermalNNEncoder expects input of shape (B, neighbor_length)")
        z = self.mlp(thermal_raw)       # (B, emb_dim)
        return self.ln(z)

class WAAMFeatureEncoder(nn.Module):
    """
    Tiny MLP for low-dim scalars -> E_S (default=32) + LayerNorm
    """
    def __init__(self, in_dim: int, emb_dim: int = 32):
        super().__init__()
        # self.mlp = nn.Sequential(
        #     nn.Linear(in_dim, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, emb_dim),
        #     nn.ReLU(),
        # )
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU()
        )
        self.ln = nn.LayerNorm(emb_dim)
        self.in_dim = in_dim
        self.emb_dim = emb_dim

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: (B, scalar_dim)
        returns: (B, emb_dim)
        """
        return self.ln(self.mlp(s))

class WAAMGRUModel(nn.Module):
    """
    Full model:
      ThermalEncoder(E_T=64) + ScalarEncoder(E_S=32)
      -> concat -> LayerNorm -> GRU(128) -> Head -> 2 outputs
    """
    def __init__(self, scalar_dim: int, use_thermal: bool = True,
                 thermal_emb: int = 64, scalar_emb: int = 32,
                 rnn_hidden: int = 128, rnn_layers: int = 1,
                 output_layers: int = 1):
        super().__init__()
        if use_thermal:
            self.thermal_enc = ThermalEncoder(emb_dim=thermal_emb)
        self.scalar_enc = WAAMFeatureEncoder(in_dim=scalar_dim, emb_dim=scalar_emb)
        if use_thermal:
            self.fuse_ln = nn.LayerNorm(thermal_emb + scalar_emb)
        else:
            self.fuse_ln = nn.LayerNorm(scalar_emb)
        
        if use_thermal:
            self.rnn = nn.GRU(input_size=thermal_emb + scalar_emb,
                            hidden_size=rnn_hidden,
                            num_layers=rnn_layers,
                            batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=scalar_emb,
                            hidden_size=rnn_hidden,
                            num_layers=rnn_layers,
                            batch_first=True)

        self.head = nn.Sequential()
        for _ in range(output_layers - 1):
            self.head.append(nn.Linear(rnn_hidden, rnn_hidden))
            self.head.append(nn.ReLU())
        self.head.append(nn.Linear(rnn_hidden, 2))     # height, width

        self.use_thermal = use_thermal
    
    def forward(self,
                scalars_seq: torch.Tensor,     # (B, T, scalar_dim)
                thermals_seq: torch.Tensor,    # (B, T, neighbor_length)
                lengths: torch.Tensor          # (B,) true lengths
                ) -> torch.Tensor:
        """
        Returns predictions for all padded steps: (B, T, 2)
        You should mask loss using 'lengths' outside.
        """
        B, T, _ = scalars_seq.shape

        # Encode per time step in batch mode:
        # Flatten batch*time for encoders, then reshape back.
        S = scalars_seq.reshape(B*T, -1)          # (B*T, scalar_dim)
        Th = thermals_seq.reshape(B*T, -1)        # (B*T, neighbor_length)

        if self.use_thermal:
            e_t = self.thermal_enc(Th)                # (B*T, E_T)
        s_t = self.scalar_enc(S)                  # (B*T, E_S)
        if self.use_thermal:
            fused = torch.cat([e_t, s_t], dim=-1)     # (B*T, E_T+E_S)
        else:
            fused = s_t                               # (B*T, E_S)

        fused = self.fuse_ln(fused).reshape(B, T, -1)  # (B, T, E)

        # RNN over time
        # rnn_out, _ = self.rnn(fused)              # (B, T, H)
        # Pack -> RNN -> Unpack (sort by length desc required)
        lengths_sorted, sort_idx = lengths.sort(descending=True)
        fused_sorted = fused.index_select(0, sort_idx)

        packed = pack_padded_sequence(fused_sorted, lengths_sorted.cpu(),
                                      batch_first=True, enforce_sorted=True)
        packed_out, _ = self.rnn(packed)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # restore original order and pad to max len
        inv_idx = sort_idx.argsort()
        rnn_out = rnn_out.index_select(0, inv_idx)

        out = self.head(rnn_out)                  # (B, T, 2)
        # If some sequences are shorter than the batch max, pad head output to T (no extra compute in RNN)
        if out.size(1) < T:
            pad = out.new_zeros(B, T - out.size(1), out.size(2))
            out = torch.cat([out, pad], dim=1)
        return out

class WAAMNNModel(nn.Module):
    """
    Full model:
      ThermalEncoder(E_T=64) + ScalarEncoder(E_S=32)
      -> concat -> LayerNorm -> MLP -> 2 outputs
    """
    def __init__(self, scalar_dim: int, use_thermal: bool = True,
                 thermal_emb: int = 64, scalar_emb: int = 32,
                 nn_hidden: int = 128, nn_layers: int = 1):
        super().__init__()
        if use_thermal:
            # self.thermal_enc = ThermalEncoder(emb_dim=thermal_emb)
            # self.thermal_enc = ThermalThinEncoder(emb_dim=thermal_emb)
            self.thermal_enc = ThermalNNEncoder(emb_dim=thermal_emb)
        self.scalar_enc = WAAMFeatureEncoder(in_dim=scalar_dim, emb_dim=scalar_emb)
        if use_thermal:
            self.fuse_ln = nn.LayerNorm(thermal_emb + scalar_emb)
        else:
            self.fuse_ln = nn.LayerNorm(scalar_emb)

        self.head = nn.Sequential(nn.Linear(thermal_emb + scalar_emb if use_thermal else scalar_emb, nn_hidden),
                                  nn.ReLU())
        for _ in range(nn_layers):
            self.head.append(nn.Linear(nn_hidden, nn_hidden))
            self.head.append(nn.ReLU())
        self.head.append(nn.Linear(nn_hidden, 2))     # height, width

        self.use_thermal = use_thermal
    
    def forward(self,
                scalars_seq: torch.Tensor,     # (B, T, scalar_dim)
                thermals_seq: torch.Tensor,    # (B, T, neighbor_length)
                ) -> torch.Tensor:
        """
        Returns predictions : (B, 2)
        """

        if self.use_thermal:
            e_t = self.thermal_enc(thermals_seq)                # (B, E_T)
        s_t = self.scalar_enc(scalars_seq)                  # (B, E_S)
        if self.use_thermal:
            fused = torch.cat([e_t, s_t], dim=-1)     # (B, E_T+E_S)
        else:
            fused = s_t                               # (B, E_S)

        fused = self.fuse_ln(fused)  # (B, E)

        out = self.head(fused) 
        return out