import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import pickle
from general_robotics_toolbox import *
from PH_interp import *
from calib_analytic_grad import *
import datetime
import time
from pathlib import Path
import yaml

import sys
sys.path.append('../toolbox/')
from robot_def import *

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

def train(training_q, training_T_data, testing_q, testing_T_data,robot,param_nominal):

    print(np.degrees(training_q).shape)
    print(training_T_data.shape)
    print(np.degrees(testing_q).shape)
    print(testing_T_data.shape)

    # data preprocessing
    inputs_q2q3 = []
    training_T = []
    test_inputs_q2q3 = []
    testing_T = []
    for (q,T) in zip(training_q,training_T_data):
        inputs_q2q3.append(torch.tensor([q[1],q[2]], dtype=torch.float32))
        training_T.append(Transform(q2R(T[3:]),T[:3]))
    for (test_q,test_T) in zip(testing_q,testing_T_data):
        test_inputs_q2q3.append(torch.tensor([test_q[1],test_q[2]], dtype=torch.float32))
        testing_T.append(Transform(q2R(test_T[3:]),test_T[:3]))
    print(len(inputs_q2q3))
    print(len(test_inputs_q2q3))

    inputs_q2q3 = torch.stack(inputs_q2q3)
    test_inputs_q2q3 = torch.stack(test_inputs_q2q3)

    # Define the input size, hidden size, and output size
    input_size = 2
    hidden_sizes = [200,200,200]
    output_size = 33

    # model type
    modelType = 'NN' # 'Fourier' or 'NN' or 'FourierNN'

    # Create an instance of the neural network
    model = NeuralNetwork(input_size, output_size, hidden_sizes=hidden_sizes)

    # Create an instance of the neural network
    if modelType == 'NN':
        model = NeuralNetwork(input_size, output_size, hidden_sizes=hidden_sizes)
    elif modelType == 'Fourier':
        model = FourierNetwork(input_size, output_size)
        # load the weights from the inverse model
        weights_from_inv = np.load('PH_NN_results/FBF_Basis_Coeff.npy')
        model.output.weight.data = torch.tensor(weights_from_inv[:,:-1], dtype=torch.float32)
        model.output.bias.data = torch.tensor(weights_from_inv[:,-1], dtype=torch.float32)
        print('Loaded weights from the inverse model')
    elif modelType == 'FourierNN':
        model = NeuralFourierNetwork(input_size, output_size, hidden_sizes=hidden_sizes)
    else:
        print('Invalid model type')
        exit()

    # read model from previous trained
    # print("Load model from previous trained")
    model.load_state_dict(torch.load('PH_NN_results/train_200_200_200_lr0.02_2409171041/best_testing_model.pt',weights_only=True))
    
    # Print the model architecture
    print(model)

    # Define the loss function
    loss_fn = TransformationLoss()

    # weights = torch.tensor([1]*33, dtype=torch.float32)
    weights_pos = 1
    weights_ori = 180/np.pi 
    # weights_ori = 1

    # statistics before training
    model.eval()
    test_outputs = model(test_inputs_q2q3)
    test_loss, test_p_error_all, test_ori_error_all = loss_fn(test_outputs, testing_T, testing_q, robot, param_nominal, weights_pos, weights_ori)
    test_p_error_norm_all = np.linalg.norm(test_p_error_all,axis=1)
    print('Before training:')
    print(f'Testing error: mean={np.mean(test_p_error_norm_all):.4f}, max={np.max(test_p_error_norm_all):.4f}')
    
    # Define the learning rate
    learning_rate = 0.0001
    # Define the number of epochs
    num_epochs = 1005
    # Define the optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # or
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # get save folder path
    formatted_string = datetime.datetime.now().strftime("%Y%m%d%H%M")
    formatted_string = formatted_string[2:]
    folder_path = 'PH_NN_results/trainDirect_'
    if modelType != 'Fourier':
        for h in hidden_sizes:
            folder_path += str(h)+'_'
    folder_path += modelType+'_'
    folder_path += 'lr'+str(learning_rate)+'_'
    folder_path += 'wp'+str(round(weights_pos,2))+'_'
    folder_path += 'wo'+str(round(weights_ori,2))+'_'
    folder_path += formatted_string+'/'
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    # save a training parameters meta yaml file to folder_path
    meta_data = {'input_size': input_size, 'hidden_sizes': hidden_sizes, 'output_size': output_size, 'learning_rate': learning_rate, 'num_epochs': num_epochs}
    meta_data['modelType'] = modelType
    meta_data['weighted'] = True
    meta_data['weights_pos'] = weights_pos
    meta_data['weights_ori'] = weights_ori
    with open(folder_path+'meta_data.yaml', 'w') as file:
        documents = yaml.dump(meta_data, file)

    # Training loop
    loss_all = []
    test_loss_all = []
    training_mean_error_all = []
    testing_mean_error_all = []
    training_max_error_all = []
    testing_max_error_all = []
    data_sample_epoches = []
    best_loss = 1e10
    best_training_error = 1e10
    best_testing_error = 1e10

    print("Start training")
    training_start_time = time.time()
    training_t_epoch = []
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Forward pass
        model.train()
        optimizer.zero_grad() # set the gradients to zero
        outputs = model(inputs_q2q3) # get the output
        # Compute loss
        loss, p_error_all, ori_error_all = loss_fn(outputs, training_T, training_q, robot, param_nominal, weights_pos, weights_ori)
        p_error_norm_all = np.linalg.norm(p_error_all,axis=1)

        # Backward pass and optimization
        loss.backward() # compute the gradients
        optimizer.step() # update the weights

        # get testing data loss
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_inputs_q2q3)
            test_loss, test_p_error_all, test_ori_error_all = loss_fn(test_outputs, testing_T, testing_q, robot, param_nominal, weights_pos, weights_ori)
            test_p_error_norm_all = np.linalg.norm(test_p_error_all,axis=1)

        # Print the loss for every N epochs
        print_loss = True
        # if epoch<101:
        #     if (epoch+1) % 1 == 0:
        #         print_loss = True
        # else:
        #     if (epoch+1) % 100 == 0:
        #         print_loss = True
        if print_loss:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
            print(f'Train error: mean={np.mean(p_error_norm_all):.4f}, max={np.max(p_error_norm_all):.4f}')
            print(f'Test error: mean={np.mean(test_p_error_norm_all):.4f}, max={np.max(test_p_error_norm_all):.4f}')
        # save model if the test loss is the best
        if best_loss > test_loss.item():
            best_loss = test_loss.item()
            torch.save(model.state_dict(), folder_path+'best_lost_model.pt')
        loss_all.append(loss.item())
        test_loss_all.append(test_loss.item())
        np.save(folder_path+'loss_all.npy',np.array(loss_all)) # save the loss
        np.save(folder_path+'test_loss_all.npy',np.array(test_loss_all)) # save the test loss
        # save the model if the training error is the best
        if best_training_error > np.max(p_error_norm_all):
            best_training_error = np.max(p_error_norm_all)
            torch.save(model.state_dict(), folder_path+'best_training_model.pt')
        # save the model if the testing error is the best
        if best_testing_error > np.max(test_p_error_norm_all):
            best_testing_error = np.max(test_p_error_norm_all)
            torch.save(model.state_dict(), folder_path+'best_testing_model.pt')
        # save the training and testing position error
        training_mean_error_all.append(np.mean(p_error_norm_all))
        testing_mean_error_all.append(np.mean(test_p_error_norm_all))
        training_max_error_all.append(np.max(p_error_norm_all))
        testing_max_error_all.append(np.max(test_p_error_norm_all))
        data_sample_epoches.append(epoch)
        np.save(folder_path+'training_mean_error_all.npy',np.array(training_mean_error_all))
        np.save(folder_path+'testing_mean_error_all.npy',np.array(testing_mean_error_all))
        np.save(folder_path+'training_max_error_all.npy',np.array(training_max_error_all))
        np.save(folder_path+'testing_max_error_all.npy',np.array(testing_max_error_all))
        np.save(folder_path+'data_sample_epoches.npy',np.array(data_sample_epoches))

        # training time for each epoch
        epoch_end_time = time.time()
        training_t_epoch.append(epoch_end_time-epoch_start_time)
        print(f'Mean epoch time: {np.mean(training_t_epoch):.2f}, Total time: {epoch_end_time-training_start_time:.2f}')
            
    print('Training time:',time.time()-training_start_time)            

Rx=np.array([1,0,0])
Ry=np.array([0,1,0])
Rz=np.array([0,0,1])

ph_dataset_date='0801'
test_dataset_date='0801'
config_dir='../config/'

robot_type = 'R1'

if robot_type == 'R1':
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
elif robot_type == 'R2':
    robot_marker_dir=config_dir+'MA1440_marker_config/'
    tool_marker_dir=config_dir+'mti_marker_config/'
    robot=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',\
                        tool_file_path=config_dir+'mti.csv',\
                        pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
                        base_marker_config_file=robot_marker_dir+'MA1440_'+ph_dataset_date+'_marker_config.yaml',\
                        tool_marker_config_file=tool_marker_dir+'mti_'+ph_dataset_date+'_marker_config.yaml')
    nom_P=np.array([[0,0,0],[155,0,0],[0,0,614],\
                   [640,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
    nom_H=np.array([[0,0,1],[0,1,0],[0,-1,0],\
                   [-1,0,0],[0,-1,0],[-1,0,0]]).T

# T_base_basemarker = robot.T_base_basemarker
# T_basemarker_base = T_base_basemarker.inv()
robot.P_nominal=deepcopy(robot.robot.P)
robot.H_nominal=deepcopy(robot.robot.H)
robot.P_nominal=robot.P_nominal.T
robot.H_nominal=robot.H_nominal.T
robot = get_H_param_axis(robot) # get the axis to parametrize H
param_nominal = np.array(np.reshape(robot.robot.P.T,-1).tolist()+[0]*12)

#### using rigid body
use_toolmaker=True
T_base_basemarker = robot.T_base_basemarker
T_basemarker_base = T_base_basemarker.inv()

if use_toolmaker:
    robot.robot.R_tool = robot.T_toolmarker_flange.R
    robot.robot.p_tool = robot.T_toolmarker_flange.p
    robot.T_tool_toolmarker = Transform(np.eye(3),[0,0,0])
    
    # robot.robot.R_tool = np.eye(3)
    # robot.robot.p_tool = np.zeros(3)
    # robot.T_tool_toolmarker = robot.T_toolmarker_flange.inv()

PH_data_dir='PH_grad_data/test'+ph_dataset_date+'_'+robot_type+'/train_data_'
# test_data_dir='kinematic_raw_data/test'+test_dataset_date+'_aftercalib/'
test_data_dir='kinematic_raw_data/test'+test_dataset_date+'_'+robot_type+'/'

print(PH_data_dir)
print(test_data_dir)

use_raw=False
test_robot_q = np.loadtxt(test_data_dir+'robot_q_align.csv',delimiter=',')
test_mocap_T = np.loadtxt(test_data_dir+'mocap_T_align.csv',delimiter=',')

train_robot_q = np.loadtxt(PH_data_dir+'robot_q_align.csv',delimiter=',')
train_mocap_T = np.loadtxt(PH_data_dir+'mocap_T_align.csv',delimiter=',')

# randomly choose half of test data as training data
np.random.seed(0)
rand_index = np.random.permutation(len(test_robot_q))
train_robot_q = np.vstack((train_robot_q,test_robot_q[rand_index[:int(len(rand_index)/2)]]))
train_mocap_T = np.vstack((train_mocap_T,test_mocap_T[rand_index[:int(len(rand_index)/2)]]))
# test_robot_q = test_robot_q[rand_index[int(len(rand_index)/2):]]
# test_mocap_T = test_mocap_T[rand_index[int(len(rand_index)/2):]]

train(train_robot_q,train_mocap_T,test_robot_q,test_mocap_T,robot,param_nominal)

# split_index = len(train_robot_q)
# test_robot_q = np.vstack((train_robot_q,test_robot_q))
# test_mocap_T = np.vstack((train_mocap_T,test_mocap_T))

# calib_file_name = 'calib_PH_q_ana.pickle'
# with open(PH_data_dir+calib_file_name,'rb') as file:
#     PH_q=pickle.load(file)

# # ph_param_fbf=PH_Param(nom_P,nom_H)
# # ph_param_fbf.fit(PH_q,method='FBF')

# # get theta phi
# train_q=[]
# param_PH_q = []
# for qkey in PH_q.keys():
#     # NN data input: q2 q3
#     train_q.append(np.array(qkey))
#     # NN output: P H
#     this_H = PH_q[qkey]['H']
#     param_H = []
#     for i,h in enumerate(this_H.T):
#         theta_sol = subproblem2(nom_H[:,i], h, robot.param_k2[i], robot.param_k1[i])
#         theta_sol = theta_sol[0] if theta_sol[0][0]<np.pi/2 and theta_sol[0][0]>-np.pi/2 else theta_sol[1]
#         param_H.extend(theta_sol[::-1])
#     param_PH = np.array(np.reshape(PH_q[qkey]['P'].T,-1).tolist()+param_H)
#     param_PH_q.append(param_PH-param_nominal) # relative to nominal, predict the difference

# ## NN input: training q, 2x1
# ## NN output: training param_PH, 33x1
# ## train the NN
# train(np.array(train_q),np.array(param_PH_q),train_robot_q,train_mocap_T,test_robot_q,test_mocap_T,robot,param_nominal)


        
