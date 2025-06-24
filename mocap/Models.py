import torch
import torch.nn as nn
from torch.autograd import Function
from calib_analytic_grad import *
from robotics_utils import *

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


# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    # def forward(self, x):
    #     h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
    #     c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
    #     out, _ = self.lstm(x, (h0, c0))
    #     out = self.fc(out[:, -1, :])  # Get the last time step's output
    #     return out

    def forward(self, x_true, u):
        batch_size, seq_len, _ = x_true.size()
        device = x_true.device

        # Initial hidden states
        h_t = torch.zeros(batch_size, self.hidden_size, device=device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=device)

        y_pred = x_true[:, 0, :]  # initial prediction using ground truth at t=0
        predictions = []

        for t in range(seq_len - 1):
            error = x_true[:, t, :] - y_pred
            u_t = u[:, t, :]
            input_t = torch.cat([error, u_t], dim=1)  # (batch, 4)

            h_t, c_t = self.lstm_cell(input_t, (h_t, c_t))
            y_pred = self.fc(h_t)
            predictions.append(y_pred)

        predictions = torch.stack(predictions, dim=1)
        return predictions