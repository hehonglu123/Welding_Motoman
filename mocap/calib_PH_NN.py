import numpy as np
import torch
import torch.nn as nn
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

def test_fwd_accuracy(model, data_q, data_T,robot,param_nominal):
    
    p_error_all = []
    for i,q in enumerate(data_q):
        q2q3 = np.array([q[1],q[2]])
        q2q3 = torch.tensor(q2q3, dtype=torch.float32)
        pred_PH = model(q2q3)
        pred_PH = pred_PH.detach().numpy() + param_nominal
        robot = get_PH_from_param(pred_PH,robot,unit='radians')
        T_pred = robot.fwd(q)
        p_error = np.linalg.norm(T_pred.p - data_T[i][:3])
        p_error_all.append(p_error)
    return p_error_all


def train(inputs_q2q3, targets_PH, training_q, training_T, testing_q, testing_T,robot,param_nominal,robot_type):

    print(np.degrees(inputs_q2q3).shape)
    print(targets_PH.shape)

    # data preprocessing
    inputs_q2q3 = torch.tensor(inputs_q2q3, dtype=torch.float32)
    targets_PH = torch.tensor(targets_PH, dtype=torch.float32)

    # Define the input size, hidden size, and output size
    input_size = 2
    hidden_sizes = [400,400]
    output_size = 33

    # Create an instance of the neural network
    model = NeuralNetwork(input_size, output_size, hidden_sizes=hidden_sizes)

    # Print the model architecture
    print(model)
    # Define the loss function
    loss_fn = nn.MSELoss()
    weighted = False
    if weighted:
        loss_fn = WeightedMSELoss()
        # weights = torch.tensor([1]*33, dtype=torch.float32)
        weights_P = 1
        weights_H = 180/np.pi
        weights = torch.tensor(np.append(np.ones(21)*weights_P,np.ones(12)*weights_H), dtype=torch.float32)
    # Define the learning rate
    learning_rate = 0.00001
    # Define the number of epochs
    num_epochs = 100000
    # Define the optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # or
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # get save folder path
    formatted_string = datetime.datetime.now().strftime("%Y%m%d%H%M")
    formatted_string = formatted_string[2:]
    folder_path = 'PH_NN_results/train_'
    folder_path += robot_type+'_'
    for h in hidden_sizes:
        folder_path += str(h)+'_'
    folder_path += 'lr'+str(learning_rate)+'_'
    if weighted:
        folder_path += 'weighted_'
    folder_path += formatted_string+'/'
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    # save a training parameters meta yaml file to folder_path
    meta_data = {'input_size': input_size, 'hidden_sizes': hidden_sizes, 'output_size': output_size, 'learning_rate': learning_rate, 'num_epochs': num_epochs}
    meta_data['robot_type'] = robot_type
    meta_data['weighted'] = weighted
    if weighted:
        meta_data['weights_P'] = weights_P
        meta_data['weights_H'] = weights_H
    with open(folder_path+'meta_data.yaml', 'w') as file:
        documents = yaml.dump(meta_data, file)

    # Training loop
    loss_all = []
    training_mean_error_all = []
    testing_mean_error_all = []
    training_max_error_all = []
    testing_max_error_all = []
    data_sample_epoches = []
    best_loss = 1e10
    best_training_error = 1e10
    best_testing_error = 1e10

    training_start_time = time.time()
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(inputs_q2q3)
        if weighted:
            loss = loss_fn(outputs, targets_PH, weights)
        else:
            loss = loss_fn(outputs, targets_PH)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get testing data loss
        # test_outputs = model(test_inputs_q2q3)
        # test_loss = loss_fn(test_outputs, test_targets_PH)

        # Print the loss for every 10 epochs
        print_loss = False
        print_error = False
        if epoch<1001:
            if (epoch+1) % 10 == 0:
                print_loss = True
            if (epoch+1) % 100 == 0:
                print_error = True
        else:
            if (epoch+1) % 100 == 0:
                print_loss = True
            if (epoch+1) % 500 == 0:
                print_error = True

        if best_loss > loss.item():
            best_loss = loss.item()
            torch.save(model.state_dict(), folder_path+'best_lost_model.pt')
        loss_all.append(loss.item())
        np.save(folder_path+'loss_all.npy',np.array(loss_all)) # save the loss
        if print_loss:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        if print_error:
            training_T_error = test_fwd_accuracy(model, training_q, training_T,robot,param_nominal)
            testing_T_error = test_fwd_accuracy(model, testing_q, testing_T,robot,param_nominal)
            # print training and testing error, mean, max
            print(f'Training error: mean={np.mean(training_T_error):.4f}, max={np.max(training_T_error):.4f}')
            print(f'Testing error: mean={np.mean(testing_T_error):.4f}, max={np.max(testing_T_error):.4f}')
            training_mean_error_all.append(np.mean(training_T_error))
            testing_mean_error_all.append(np.mean(testing_T_error))
            training_max_error_all.append(np.max(training_T_error))
            testing_max_error_all.append(np.max(testing_T_error))
            data_sample_epoches.append(epoch)
            # save the model
            if best_training_error > np.max(training_T_error):
                best_training_error = np.max(training_T_error)
                torch.save(model.state_dict(), folder_path+'best_training_model.pt')
            if best_testing_error > np.max(testing_T_error):
                best_testing_error = np.max(testing_T_error)
                torch.save(model.state_dict(), folder_path+'best_testing_model.pt')
            np.save(folder_path+'training_mean_error_all.npy',np.array(training_mean_error_all))
            np.save(folder_path+'testing_mean_error_all.npy',np.array(testing_mean_error_all))
            np.save(folder_path+'training_max_error_all.npy',np.array(training_max_error_all))
            np.save(folder_path+'testing_max_error_all.npy',np.array(testing_max_error_all))
            np.save(folder_path+'data_sample_epoches.npy',np.array(data_sample_epoches))
    print('Training time:',time.time()-training_start_time)            

Rx=np.array([1,0,0])
Ry=np.array([0,1,0])
Rz=np.array([0,0,1])

ph_dataset_date='0804'
test_dataset_date='0804'
config_dir='../config/'

robot_type = 'R2'

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

# split_index = len(train_robot_q)
# test_robot_q = np.vstack((train_robot_q,test_robot_q))
# test_mocap_T = np.vstack((train_mocap_T,test_mocap_T))

calib_file_name = 'calib_PH_q_ana.pickle'
with open(PH_data_dir+calib_file_name,'rb') as file:
    PH_q=pickle.load(file)

# ph_param_fbf=PH_Param(nom_P,nom_H)
# ph_param_fbf.fit(PH_q,method='FBF')

# get theta phi
train_q=[]
param_PH_q = []
for qkey in PH_q.keys():
    # NN data input: q2 q3
    train_q.append(np.array(qkey))
    # NN output: P H
    this_H = PH_q[qkey]['H']
    param_H = []
    for i,h in enumerate(this_H.T):
        theta_sol = subproblem2(nom_H[:,i], h, robot.param_k2[i], robot.param_k1[i])
        theta_sol = theta_sol[0] if theta_sol[0][0]<np.pi/2 and theta_sol[0][0]>-np.pi/2 else theta_sol[1]
        param_H.extend(theta_sol[::-1])
    param_PH = np.array(np.reshape(PH_q[qkey]['P'].T,-1).tolist()+param_H)
    param_PH_q.append(param_PH-param_nominal) # relative to nominal, predict the difference

## NN input: training q, 2x1
## NN output: training param_PH, 33x1
## train the NN
train(np.array(train_q),np.array(param_PH_q),train_robot_q,train_mocap_T,test_robot_q,test_mocap_T,robot,param_nominal,robot_type)


        
