# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, target

class HoughNet(nn.Module):
    def __init__(self):
        super(HoughNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 6)  # Output: npoints, a1, a2, a3, b1, b2

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool3d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool3d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.unsqueeze(1).float()
            targets = targets.float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

def main():
    # we need to load our data here
    # data = np.random.rand(100, 33, 33, 33)
    # targets = np.random.rand(100, 6)

    transform = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
    dataset = CustomDataset(data, targets, transform=transform)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = HoughNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, criterion, optimizer, num_epochs=25)

if __name__ == "__main__":
    main()
