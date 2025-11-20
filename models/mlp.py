import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model for MNIST classification.
    Serves as a baseline model.
    """
    def __init__(self):
        super(MLP, self).__init__()
        # Define the layers of the MLP
        # Input layer: 28*28 = 784 features (flattened MNIST image)
        # Hidden layer 1: 128 neurons
        # Hidden layer 2: 64 neurons
        # Output layer: 10 neurons (for 10 digits 0-9)
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x (torch.Tensor): Input tensor (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output tensor (batch_size, 10).
        """
        # Flatten the input image from (batch_size, 1, 28, 28) to (batch_size, 784)
        x = x.view(-1, 28 * 28)
        # Apply first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Apply second fully connected layer with ReLU activation
        x = F.relu(self.fc2(x))
        # Apply third fully connected layer (output layer)
        x = self.fc3(x)
        return x

# Optional: Add a simple test to verify model creation and forward pass
if __name__ == '__main__':
    # Create a dummy input tensor
    # Batch size of 1, 1 channel, 28x28 image
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # Instantiate the MLP model
    model = MLP()
    
    # Perform a forward pass
    output = model(dummy_input)
    
    print(f"MLP model output shape: {output.shape}") # Expected: [1, 10]
    print("MLP model created and tested successfully.")
