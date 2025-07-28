import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) implementation in PyTorch.
    
    Args:
        input_dim (int): Number of input features (e.g., 784 for MNIST).
        hidden_dim (int): Number of hidden units.
        output_dim (int): Number of output classes.
    """
    def __init__(self, input_dim=784, hidden_dim=512, output_dim=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
    def predict(self, x):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            outputs = self.forward(x)
            predicted = outputs.argmax(dim=1)
        return predicted