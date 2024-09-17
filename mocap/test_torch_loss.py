import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function

from calib_analytic_grad import *

import sys
sys.path.append('../toolbox/')
from robot_def import *

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

# Custom transformation error with manually specified gradient, the analytical gradient using autograd.Function
class TransformationLossFunction(Function):
    @staticmethod
    def forward(ctx, predict_PH, target, robot, joint_angles, weight_pos=1, weight_ori=1):

        error_all = []
        for i,(q,ph,T) in enumerate(zip(joint_angles,predict_PH,target)):
            robot = get_PH_from_param(ph.detach().numpy(),robot,unit='radians')
            T_pred = robot.fwd(q)
            p_error = np.linalg.norm(T_pred.p - T.p)
            omega_d= np.linalg.norm(s_err_func(T_pred.R@T.R.T))
            error_all.append(weight_pos*p_error + weight_ori*omega_d)
        loss = torch.tensor(np.mean(error_all))

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors from the forward pass
        input, target = ctx.saved_tensors
        # Manually compute the gradient: grad_input = input - target
        grad_input = input - target
        # Apply the chain rule: multiply by grad_output
        return grad_input * grad_output, None

# Custom loss class that inherits from nn.Module
class TransformationLoss(nn.Module):
    def __init__(self):
        super(TransformationLoss, self).__init__()

    def forward(self, input, target):
        # Use the custom autograd function for the forward pass
        return TransformationLossFunction.apply(input, target)
    

def test_FWD_loss():
    
    # defined robot
    ph_dataset_date='0801'
    test_dataset_date='0801'
    config_dir='../config/'
    robot_type = 'R1'
    robot_marker_dir=config_dir+'MA2010_marker_config/'
    tool_marker_dir=config_dir+'weldgun_marker_config/'
    robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',\
                        tool_file_path=config_dir+'torch.csv',d=15,\
                        #  tool_file_path='',d=0,\
                        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
                        base_marker_config_file=robot_marker_dir+'MA2010_'+ph_dataset_date+'_marker_config.yaml',\
                        tool_marker_config_file=tool_marker_dir+'weldgun_'+ph_dataset_date+'_marker_config.yaml')
    nom_P=np.array([[0,0,0],[150,0,0],[0,0,760],\
                   [1082,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
    nom_H=np.array([[0,0,1],[0,1,0],[0,-1,0],\
                   [-1,0,0],[0,-1,0],[-1,0,0]]).T
    # get data
    PH_data_dir='PH_grad_data/test'+ph_dataset_date+'_'+robot_type+'/train_data_'
    train_robot_q = np.loadtxt(PH_data_dir+'robot_q_align.csv',delimiter=',')
    train_mocap_T = np.loadtxt(PH_data_dir+'mocap_T_align.csv',delimiter=',')

    # randomize a tensor with size N(training data size) x 33, between 0 to 0.1
    predict_PH = torch.rand((train_robot_q.shape[0],33))*0.1

def test_weighted_MSE():
    # Example data: batch of 3 samples, each with 4 output elements
    input = torch.tensor([[2.5, 0.5, 2.0, 1.0],
                          [1.5, 1.0, 3.0, 0.5],
                          [0.0, 2.0, 1.0, 1.5]], requires_grad=True)
    
    target = torch.tensor([[3.0, 0.0, 1.5, 1.2],
                           [1.0, 0.8, 2.5, 0.0],
                           [0.5, 1.5, 1.2, 1.3]])
    
    # Weights for each element of the output (per output element dimension)
    weights = torch.tensor([0.7, 0.1, 0.2, 0.5])

    # Initialize the loss function
    criterion = WeightedMSELoss()

    # Compute the loss
    loss = criterion(input, target, weights)

    # Backpropagate
    loss.backward()

    print(f'Loss: {loss.item()}')

    ## verify the loss with numpy
    input_np = input.detach().numpy()
    target_np = target.detach().numpy()

    diff = input_np - target_np
    squared_diff = diff ** 2
    weighted_squared_diff = squared_diff * weights.numpy()
    print(weighted_squared_diff)

# Example usage
if __name__ == "__main__":
    # test_weighted_MSE()
    test_FWD_loss()