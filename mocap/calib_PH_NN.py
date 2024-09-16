import numpy as np
import torch
import torch.nn as nn
import pickle
from general_robotics_toolbox import *
from PH_interp import *
from calib_analytic_grad import *

import sys
sys.path.append('../toolbox/')
from robot_def import *

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

def train():

    # Define the input size, hidden size, and output size
    input_size = 2
    hidden_sizes = [20,20]
    output_size = 33

    # Create an instance of the neural network
    model = NeuralNetwork(input_size, output_size, hidden_sizes=hidden_sizes)

    # Print the model architecture
    print(model)

    # Define the loss function
    loss_fn = nn.MSELoss()

    # Define the learning rate
    learning_rate = 0.001

    # Define the number of epochs
    num_epochs = 100

    # Define the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # or
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get testing data loss
        test_outputs = model(test_inputs)
        test_loss = loss_fn(test_outputs, test_targets)

        # Print the loss for every 10 epochs
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

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

split_index = len(train_robot_q)
test_robot_q = np.vstack((train_robot_q,test_robot_q))
test_mocap_T = np.vstack((train_mocap_T,test_mocap_T))

calib_file_name = 'calib_PH_q_ana.pickle'
print(calib_file_name)
with open(PH_data_dir+calib_file_name,'rb') as file:
    PH_q=pickle.load(file)

# ph_param_fbf=PH_Param(nom_P,nom_H)
# ph_param_fbf.fit(PH_q,method='FBF')

# get theta phi
train_q=[]
for qkey in PH_q.keys():
    train_q.append(np.array(qkey))

    this_H = PH_q[qkey]['H']
    param_H = []
    for i,h in enumerate(this_H.T):
        theta_sol = subproblem2(nom_H[:,i], h, robot.param_k2[i], robot.param_k1[i])
        theta_sol = theta_sol[0] if theta_sol[0][0]<np.pi/2 and theta_sol[0][0]>-np.pi/2 else theta_sol[1]
        param_H.extend(theta_sol[::-1])
    param_PH = np.array(np.reshape(PH_q[qkey]['P'].T,-1).tolist()+param_H)
    robot = get_PH_from_param(param_PH,robot)

    
        
