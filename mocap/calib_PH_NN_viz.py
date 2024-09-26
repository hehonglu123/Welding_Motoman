import torch
import torch.nn as nn
import numpy as np
import yaml

import matplotlib.pyplot as plt

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
    
# Define the input size, hidden size, and output size

# model_name = 'train_200_200_200_lr0.02_2409171041'
model_name = 'train_R1_200_200_200_Fourier_lr0.02_2409260929'

# read meta data
with open('PH_NN_results/'+model_name+'/meta_data.yaml') as file:
    meta_data = yaml.load(file, Loader=yaml.FullLoader)

input_size = 2
output_size = 33
hidden_sizes = meta_data['hidden_sizes']

# Create an instance of the neural network
if 'use_Fourier' not in meta_data or not meta_data['use_Fourier']:
    model = NeuralNetwork(input_size, output_size, hidden_sizes=hidden_sizes)
else:
    model = NeuralFourierNetwork(input_size, output_size, hidden_sizes=hidden_sizes)
# read model from previous trained
# print("Load model from previous trained")

model.load_state_dict(torch.load('PH_NN_results/'+model_name+'/best_testing_model.pt',weights_only=True))

# Extract the weights of the output layer
output_layer_weights = model.output.weight.data.cpu().numpy()
output_layer_bias = model.output.bias.data.cpu().numpy()
print(output_layer_weights.shape)
print(output_layer_bias.shape)

output_layer_weights_bias = np.concatenate((output_layer_bias.reshape(-1,1),output_layer_weights),axis=1)

# Plot the weights of the last hidden layer
plt.figure(figsize=(10, 6))
plt.imshow(np.fabs(output_layer_weights_bias), aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Weights of the Last Hidden Layer')
plt.xlabel('Neuron Index')
plt.ylabel('Weight Index')
plt.tight_layout()
plt.show()

# singular value decomposition of weights
U, S, V = np.linalg.svd(output_layer_weights, full_matrices=False)
# plot singular values
plt.figure()
plt.plot(S, 'o-')
plt.title('Singular Values of the Output Layer Weights')
plt.xlabel('Singular Value Index')
plt.ylabel('Singular Value')
plt.grid()
plt.show()
for v in V.T:
    print(np.linalg.norm(v))
# plot V matrix
plt.figure(figsize=(10, 6))
plt.imshow(np.fabs(V), aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('V Matrix of the Output Layer Weights')
plt.xlabel('Neuron Index')
plt.ylabel('Singular Value Index')
plt.tight_layout()
plt.show()