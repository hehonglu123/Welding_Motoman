import torch

import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, outputs, targets):
        # Calculate the custom loss
        loss = torch.mean(torch.abs(outputs - targets))

        # Calculate the gradient manually
        grad = torch.sign(outputs - targets)

        return loss, grad

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

# Define the input size, hidden size, and output size
input_size = 10
hidden_sizes = [20,20]
output_size = 1

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