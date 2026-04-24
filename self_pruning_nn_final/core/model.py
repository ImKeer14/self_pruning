import torch
import torch.nn as nn
import torch.nn.functional as F

class PrunableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.gate_scores = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores * 5.0)
        pruned_weights = self.weight * gates
        return F.conv2d(x, pruned_weights, self.bias)

class SelfPruningCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = PrunableConv(3, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = PrunableConv(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
