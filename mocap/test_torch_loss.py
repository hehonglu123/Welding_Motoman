import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function

from calib_analytic_grad import *

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
            robot = get_PH_from_param(ph.numpy(),robot,unit='radians')
            T_pred = robot.fwd(q)
            p_error = np.linalg.norm(T_pred.p - T.p)
            omega_d= np.linalg.norm(s_err_func(T_pred.R@T.R.T))
            error_all.append(weight_pos*p_error + weight_ori*omega_d)
        loss = np.mean(error_all)

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

# Example usage
if __name__ == "__main__":
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