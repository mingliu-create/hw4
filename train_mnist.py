import os
import argparse
import torch
import torch.optim as optim

# Import our custom modules
from models.mlp import MLP
from models.cnn import CNN
from utils.data import load_mnist_data
from utils.train_utils import train, test

def main():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training Script')
    parser.add_argument('--model', type=str, default='cnn', choices=['mlp', 'cnn'],
                        help='model to train (mlp or cnn, default: cnn)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=5, # Reduced default for quicker first run
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    # --- 2. Device Setup ---
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # --- 3. Data Loading ---
    train_loader, test_loader = load_mnist_data(args.batch_size)

    # --- 4. Model Initialization ---
    if args.model.lower() == 'cnn':
        model = CNN().to(device)
    elif args.model.lower() == 'mlp':
        model = MLP().to(device)
    else:
        # This case is technically handled by argparse's `choices`
        raise ValueError("Invalid model type specified. Choose 'mlp' or 'cnn'.")
    
    print(f"Training model: {args.model.upper()}")
    
    # --- 5. Optimizer Setup ---
    # Adam is a good general-purpose optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- 6. Training Loop ---
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # --- 7. Save the Model ---
    # Ensure the directory for saved models exists
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    
    # Define the file path for the saved model
    save_path = os.path.join(save_dir, f'{args.model.lower()}_mnist.pth')
    
    # Save the model's state dictionary
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining finished. Model saved to {save_path}")

if __name__ == '__main__':
    main()
