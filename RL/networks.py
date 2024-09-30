import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim) 

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = F.relu((self.fc1(x)))
        x = F.relu((self.fc2(x)))
        x = self.fc3(x)
        return x


class BasicCNN(nn.Module):
    def __init__(self, input_dim, input_channels, output_dim, conv3=False):
        super().__init__()
        h = input_dim[0]
        w = input_dim[1]
        flat = int(32 * ((h // 4 - 1) // 2 - 1) * ((w // 4 - 1) // 2 - 1))

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        if conv3:
            self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        else:
            self.conv3 = None
        self.fc1 = nn.Linear(flat, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.conv3:
            x = F.relu(self.conv3(x))  
        x = F.relu(self.fc1(x.flatten(-3)))
        x = self.fc2(x)
        return x