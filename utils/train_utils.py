import torch
import torch.nn.functional as F

def train(model, device, train_loader, optimizer, epoch):
    """
    Performs a single training epoch.

    Args:
        model (torch.nn.Module): The neural network model.
        device (torch.device): The device (CPU or GPU) to train on.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): The optimizer to use for updating weights.
        epoch (int): Current epoch number.
    """
    model.train()  # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) # Move data to the specified device
        optimizer.zero_grad()  # Zero the gradients before running the backward pass
        output = model(data)   # Forward pass
        loss = F.cross_entropy(output, target) # Calculate the loss
        loss.backward()        # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step()       # Perform a single optimization step (parameter update)

        if batch_idx % 100 == 0: # Print training status every 100 batches
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    """
    Evaluates the model on the test dataset.

    Args:
        model (torch.nn.Module): The neural network model.
        device (torch.device): The device (CPU or GPU) to evaluate on.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.

    Returns:
        float: The test accuracy.
    """
    model.eval()   # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad(): # Disable gradient calculations during evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device) # Move data to the specified device
            output = model(data) # Forward pass
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # Count correct predictions

    test_loss /= len(test_loader.dataset) # Calculate average loss

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    return 100. * correct / len(test_loader.dataset) # Return accuracy
