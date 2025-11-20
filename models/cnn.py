import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) model for MNIST classification.
    Designed for 28x28 grayscale images.
    """
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer
        # Input channels: 1 (grayscale image)
        # Output channels: 32
        # Kernel size: 3x3
        # Padding: 1 (to keep output size same as input size for conv layer)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # Second convolutional layer
        # Input channels: 32 (from previous layer)
        # Output channels: 64
        # Kernel size: 3x3
        # Padding: 1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Max pooling layer
        # Kernel size: 2x2
        # Stride: 2 (reduces image size by half)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layer
        # After two conv + pool layers, image size becomes 7x7.
        # So, 64 channels * 7 * 7 features are flattened.
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5) # Add dropout layer
        # Output layer
        self.fc2 = nn.Linear(128, 10) # 10 classes for digits 0-9

    def forward(self, x):
        """
        Forward pass through the CNN.

        Args:
            x (torch.Tensor): Input tensor (batch_size, 1, 28, 28).

        Returns:
            torch.Tensor: Output tensor (batch_size, 10).
        """
        # Apply conv1 -> ReLU -> pool
        x = self.pool(F.relu(self.conv1(x))) # (batch_size, 32, 14, 14)
        # Apply conv2 -> ReLU -> pool
        x = self.pool(F.relu(self.conv2(x))) # (batch_size, 64, 7, 7)
        
        # Flatten the output for the fully connected layers
        # The .view(-1, ...) automatically calculates the batch size
        x = x.view(-1, 64 * 7 * 7)
        
        # Apply fc1 -> ReLU
        x = F.relu(self.fc1(x))
        x = self.dropout(x) # Apply dropout
        x = self.fc2(x)
        return x

# Optional: Add a simple test to verify model creation and forward pass
if __name__ == '__main__':
    # Create a dummy input tensor
    # Batch size of 1, 1 channel, 28x28 image
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # Instantiate the CNN model
    model = CNN()
    
    # Perform a forward pass
    output = model(dummy_input)
    
    print(f"CNN model output shape: {output.shape}") # Expected: [1, 10]
    print("CNN model created and tested successfully.")
