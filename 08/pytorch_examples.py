"""Brief examples of PyTorch usage"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Generate sample data
np.random.seed(42)
X = torch.FloatTensor(np.random.randn(1000, 1, 28, 28))  # Simulated MNIST-like data
y = torch.LongTensor(np.random.randint(0, 10, 1000))  # 10 classes

# Custom dataset
class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create data loader
dataset = SimpleDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(64 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Training loop
def train_model(model, loader, epochs=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

if __name__ == '__main__':
    model = SimpleCNN()
    train_model(model, loader)
    
    # Save model
    torch.save(model.state_dict(), 'simple_cnn.pt')
    
    # Load model
    new_model = SimpleCNN()
    new_model.load_state_dict(torch.load('simple_cnn.pt'))
