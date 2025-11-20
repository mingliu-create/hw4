import torch
from torchvision import datasets, transforms 
def load_mnist_data(batch_size=64):
    """
    Loads and preprocesses the MNIST dataset.
    Args:
    batch_size (int): The batch size for the data loaders.
   Returns:
   tuple: A tuple containing (train_loader, test_loader).
   """
     # Define transformations for the MNIST dataset
     # Convert images to PyTorch tensors and normalize them
     # Mean and standard deviation for MNIST (calculated from the dataset)
    transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,)) # Standard normalization for MNIST
     ]) 
     # Load the training dataset
    test_dataset = datasets.MNIST(
         root='./data',         # Directory to download data to
         train=False,           # Specifies test set
         download=True,         # Download if not already present
         transform=transform    # Apply defined transformations
     )

     # Create a DataLoader for the test set
     # shuffle=False for test data as order does not affect evaluation
    test_loader = torch.utils.data.DataLoader(
         test_dataset,
         batch_size=batch_size,
         shuffle=False          # No need to shuffle test data
     )

    print("MNIST data loaded successfully.")
    return train_loader, test_loader

 # Optional: Add a simple test to verify data loading when running this script directly
if __name__ == '__main__':
     train_loader, test_loader = load_mnist_data()
     print(f"Number of training batches: {len(train_loader)}")
     print(f"Number of test batches: {len(test_loader)}")
     # Get one batch and print its shape
     images, labels = next(iter(train_loader))
     print(f"Shape of one image batch: {images.shape}") # Expected: [batch_size, 1, 28, 28]     
     print(f"Shape of one label batch: {labels.shape}") # Expected: [batch_size]