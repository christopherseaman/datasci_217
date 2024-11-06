#!/usr/bin/env python3
"""
PyTorch MNIST Classification Example

This script demonstrates a complete machine learning workflow using PyTorch:
1. Data Loading and Preprocessing
2. Model Architecture Design
3. Training and Validation
4. Performance Evaluation

Dataset: MNIST Handwritten Digit Recognition
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. DATA PREPARATION
def prepare_data(batch_size=64):
    """
    Prepare MNIST dataset for training and validation
    
    Steps:
    - Define data transformations (normalization)
    - Load train and test datasets
    - Create DataLoaders for batch processing
    """
    # Normalize images to [0, 1] range and then use standard normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        transform=transform
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader

# 2. MODEL ARCHITECTURE
class MNISTClassifier(nn.Module):
    """
    Simple Neural Network for MNIST Classification
    
    Architecture:
    - Fully connected layers
    - ReLU activation
    - Log softmax output
    """
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(MNISTClassifier, self).__init__()
        
        # Layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Input: Flattened image (batch_size x 784)
        Output: Class probabilities
        """
        # Flatten the input
        x = x.view(x.size(0), -1)
        
        # First layer with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Output layer with log softmax
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 3. TRAINING PROCESS
def train_model(model, train_loader, test_loader, epochs=5):
    """
    Train and validate the model
    
    Steps:
    - Define loss function (Negative Log Likelihood)
    - Define optimizer (Adam)
    - Iterate through epochs
    - Perform training and validation
    """
    # Loss and Optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        correct = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
                
                # Get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        # Print epoch statistics
        print(f'Epoch {epoch}: '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Test Loss: {test_loss/len(test_loader):.4f}, '
              f'Test Accuracy: {100. * correct / len(test_loader.dataset):.2f}%')

def main():
    """
    Main function to orchestrate the machine learning workflow
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Prepare data
    train_loader, test_loader = prepare_data()
    
    # Initialize model
    model = MNISTClassifier()
    
    # Train the model
    train_model(model, train_loader, test_loader)

if __name__ == '__main__':
    main()
