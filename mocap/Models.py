import torch
import torch.nn as nn
from torch.autograd import Function
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
            omega_d= s_err_func(T_pred.R@T.R.T)
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
        out_0 = self.fc(h0.squeeze(0))  # shape: (batch, output_size)
        out_0 = out_0.unsqueeze(1)
        out, _ = self.lstm(u, (h0, c0))
        out = self.fc(out)  # Get the last time step's output
        out = torch.cat([out_0, out], dim=1)  # Concatenate initial output with the rest
        out = out[:, :-1, :]  # Remove the last time step to match the
        out = out.contiguous()  # Ensure the output is contiguous in memory
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

        predictions = []
        # Initial hidden states
        h_t = torch.zeros(batch_size, self.hidden_size, device=self.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=self.device)
        y_pred = self.fc(h_t)  # Initial output
        predictions.append(y_pred)
        # Initial error
        error = x_true[:, self.history_length, :] - y_pred
        

        for t in range(self.history_length,seq_len-1):
            u_t = torch.flatten(u[:, t-self.history_length+1:t, :],start_dim=1)  # (batch, history_length * input_size)
            u_t = torch.cat([u[:, t, :], u_t], dim=1)  # (batch, current_input + history_length * input_size)
            y_t = torch.flatten(x_true[:, t-self.history_length:t, :], start_dim=1)  # (batch, history_length * output_size)
            # Concatenate error and inputs
            input_t = torch.cat([y_t,u_t,error], dim=1)  # (batch, history_length * output_size + current_input + history_length * input_size + error)

            h_t, c_t = self.lstm_cell(input_t, (h_t, c_t))
            y_pred = self.fc(h_t)
            predictions.append(y_pred)

            error = x_true[:, t+1, :] - y_pred

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
        out_0 = self.fc(h0.squeeze(0))  # shape: (batch, output_size)
        out_0 = out_0.unsqueeze(1)
        out, _ = self.rnn(u, h0)
        out = self.fc(out)  # Get the last time step's output
        # Concatenate initial output with the rest
        out = torch.cat([out_0, out], dim=1)
        out = out[:, :-1, :]  # Remove the last time step to match the input sequence length
        out = out.contiguous()  # Ensure the output is contiguous in memory
        return out


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
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, history_length=0, open_loop=False, device='cpu'):
        super(RNNAutoRegressionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.history_length = history_length
        self.open_loop = open_loop
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        if num_layers > 1:
            self.rnn_cell = DeepTransitionRNNCell(input_size, hidden_size, depth=num_layers)
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = device
    
    def forward(self, x_true, u):
        batch_size, seq_len, _ = x_true.size()

        predictions = []

        # Initial hidden states
        h_t = torch.zeros(batch_size, self.hidden_size, device=self.device)
        # Initial error is x_true[:, self.history_length, :] - fc(h_t)
        y_pred = self.fc(h_t)
        predictions.append(y_pred)
        error = x_true[:, self.history_length, :] - y_pred

        for t in range(self.history_length, seq_len-1):
            if not self.open_loop:
                u_t = torch.flatten(u[:, t-self.history_length+1:t, :], start_dim=1)  # (batch, history_length * input_size)
                u_t = torch.cat([u[:, t, :], u_t], dim=1)
                y_t = torch.flatten(x_true[:, t-self.history_length:t, :], start_dim=1)  # (batch, history_length * output_size)
                # Concatenate error and inputs
                input_t = torch.cat([y_t, u_t, error], dim=1)  # (batch, history_length * output_size + current_input + history_length * input_size + error)
            else:
                input_t = u[:, t, :]  # (batch, input_size)

            h_t = self.rnn_cell(input_t, h_t)
            y_pred = self.fc(h_t)
            predictions.append(y_pred)

            error = x_true[:, t+1, :] - y_pred

        predictions = torch.stack(predictions, dim=1)
        return predictions
    
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
        out_0 = self.fc(h0.squeeze(0))  # shape: (batch, output_size)
        out_0 = out_0.unsqueeze(1)
        out, _ = self.gru(u, h0)
        out = self.fc(out)  # Get the last time step's output
        # Concatenate initial output with the rest
        out = torch.cat([out_0, out], dim=1)
        out = out[:, :-1, :]  # Remove the last time step to match the input sequence length
        out = out.contiguous()  # Ensure the output is contiguous in memory
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

        predictions = []
        # Initial hidden states
        h_t = torch.zeros(batch_size, self.hidden_size, device=self.device)
        # Initial output
        y_pred = self.fc(h_t)
        predictions.append(y_pred)
        # Initial error
        error = x_true[:, self.history_length, :] - y_pred

        for t in range(self.history_length, seq_len-1):
            u_t = torch.flatten(u[:, t-self.history_length+1:t, :], start_dim=1)  # (batch, history_length * input_size)
            u_t = torch.cat([u[:, t, :], u_t], dim=1)
            y_t = torch.flatten(x_true[:, t-self.history_length:t, :], start_dim=1)
            # Concatenate error and inputs
            input_t = torch.cat([y_t, u_t, error], dim=1) # (batch, history_length * output_size + current_input + history_length * input_size + error)

            h_t = self.gru_cell(input_t, h_t)
            y_pred = self.fc(h_t)
            predictions.append(y_pred)

            error = x_true[:, t+1, :] - y_pred

        predictions = torch.stack(predictions, dim=1)
        return predictions
