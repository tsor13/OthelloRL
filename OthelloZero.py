import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = channels,
                               out_channels = channels,
                               kernel_size = 3,
                               padding = 1)
        self.relu1 = nn.ReLU()
        # TODO right here?
        self.batchnorm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(in_channels = channels,
                               out_channels = channels,
                               kernel_size = 3,
                               padding = 1)
        self.batchnorm2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        d = self.conv1(x)
        d = self.batchnorm1(d)
        d = self.relu1(d)
        d = self.conv2(d)
        d = self.batchnorm2(d)
        # skip connection
        x = x + d
        x = self.relu2(x)
        return x

class PolicyHead(nn.Module):
    def __init__(self, in_channels):
        super(PolicyHead, self).__init__()
        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = 2,
                              kernel_size = 1)
        self.batchnorm = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8*8*2, 8*8)

    def forward(self, x, possible):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = x.reshape(-1, 8*8*2)
        x = self.fc(x)
        x = x.reshape(-1, 8*8)
        # exp
        x = torch.exp(x)
        # restrict to possible moves
        x = x.reshape(-1, 8, 8)
        # mask so only possible moves
        x = x * possible
        # softmax so true probabilities
        x = x / x.sum()
        return x

class ValueHead(nn.Module):
    def __init__(self, in_channels, hidden_size=64):
        super(ValueHead, self).__init__()
        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = 1,
                              kernel_size = 1)
        self.batchnorm = nn.BatchNorm2d(1)
        self.relu1 = nn.ReLU()
        self.fcc1 = nn.Linear(8*8, hidden_size)
        self.relu2 = nn.ReLU()
        self.fcc2 = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu1(x)
        x = x.reshape(-1, 64)
        x = self.fcc1(x)
        x = self.relu2(x)
        x = self.fcc2(x)
        x = self.tanh(x)
        return x

class OthelloZero(nn.Module):
    def __init__(self, res_layers = 40, channels = 500, in_channels = 2, hidden_size = 64):
        super(OthelloZero, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channels = in_channels,
                      out_channels = channels,
                      kernel_size = 3,
                      padding = 1),
            # nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.res_blocks = nn.ModuleList([ResBlock(channels) for i in range(res_layers)])
        self.policy_head = PolicyHead(channels)
        self.value_head = ValueHead(channels, hidden_size)

    def forward(self, x):
        # split board and possible actions
        x, possible = x[:,:2], x[:,2]

        x = self.convlayer(x)
        for block in self.res_blocks:
            x = block(x)
        p = self.policy_head(x, possible)
        v = self.value_head(x)
        return p, v
