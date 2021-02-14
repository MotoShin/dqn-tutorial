import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, output):
        super(ActorNetwork, self).__init__()
        self.conv1 = _init_conv_layer(4, 32, 8, 4)
        self.conv2 = _init_conv_layer(32, 32, 4, 2)
        self.conv3 = _init_conv_layer(32, 32, 3, 1)
        self.fc1 = _init_linear_layer(7 * 7 * 32, 200)
        self.fc2 = _init_linear_layer(200, 200)
        self.mu = _init_linear_layer(200, output, 0.0003)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.mu(x)
        return torch.tanh(x)

    @staticmethod
    def _init_conv_layer(in_channel, out_channel, kernel_size, stride):
        conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)
        # initialize the layer as supplementary information says
        f = 1 / np.sqrt(conv.weight.data.size()[0])
        return _init_torchnn(conv, f)

    @staticmethod
    def _init_linear_layer(in_channel, out_channel, f=None):
        fc = nn.Linear(in_channel, out_channel)
        if f == None:
            f = 1 / np.sqrt(fc.weight.data.size()[0])
        return _init_torchnn(fc, f)

    @staticmethod
    def _init_torchnn(torchnn, value):
        torch.nn.init.uniform_(torchnn.weght.data, -value, value)
        torch.nn.init.uniform_(torchnn.bias.data, -value, value)
        return torchnn

class CriticNetwork(nn.Module):
    def __init__(self, output):
        super(CriticNetwork, self).__init__()
        self.conv1 = _init_conv_layer(4, 32, 8, 4)
        self.conv2 = _init_conv_layer(32, 32, 4, 2)
        self.conv3 = _init_conv_layer(32, 32, 3, 1)
        self.fc1 = _init_linear_layer(7 * 7 * 32, 200)
        self.fc2 = _init_linear_layer(200, 200)
        self.action_value = _init_linear_layer(200, output, 0.0003)
        self.q = nn.Linear()

    @staticmethod
    def _init_conv_layer(in_channel, out_channel, kernel_size, stride):
        conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)
        # initialize the layer as supplementary information says
        f = 1 / np.sqrt(conv.weight.data.size()[0])
        return _init_torchnn(conv, f)

    @staticmethod
    def _init_linear_layer(in_channel, out_channel, f=None):
        fc = nn.Linear(in_channel, out_channel)
        if f == None:
            f = 1 / np.sqrt(fc.weight.data.size()[0])
        return _init_torchnn(fc, f)

    @staticmethod
    def _init_torchnn(torchnn, value):
        torch.nn.init.uniform_(torchnn.weght.data, -value, value)
        torch.nn.init.uniform_(torchnn.bias.data, -value, value)
        return torchnn
