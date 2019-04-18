import torch
import torch.nn as nn
from torch.nn import functional as F


class Conv_NN(nn.Module):
    def __init__(self):
        super(Conv_NN, self).__init__()
        # input channel, output channel, Filter Size
        self.conv1 = nn.Conv2d(1, 6, 5)
        # Filter Size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # input nodes, output nodes
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        # Change to 1 dimensional Tensor
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x