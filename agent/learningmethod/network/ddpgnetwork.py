import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, output):
        super(ActorNetwork, self).__init__()
        self.conv1 = DdpgUtility.init_conv_layer(4, 32, 8, 4)
        self.conv2 = DdpgUtility.init_conv_layer(32, 32, 4, 2)
        self.conv3 = DdpgUtility.init_conv_layer(32, 32, 3, 1)
        self.fc1 = DdpgUtility.init_linear_layer(7 * 7 * 32, 200)
        self.fc2 = DdpgUtility.init_linear_layer(200, 200)
        self.mu = DdpgUtility.init_linear_layer(200, output, 0.0003)

        self.optimizer = None

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.relu(self.fc2(x))
        x = self.mu(x)
        return torch.tanh(x)

    def init_optimizer(self, lr):
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

class CriticNetwork(nn.Module):
    def __init__(self, output):
        super(CriticNetwork, self).__init__()
        self.conv1 = DdpgUtility.init_conv_layer(4, 32, 8, 4)
        self.conv2 = DdpgUtility.init_conv_layer(32, 32, 4, 2)
        self.conv3 = DdpgUtility.init_conv_layer(32, 32, 3, 1)
        self.fc1 = DdpgUtility.init_linear_layer(7 * 7 * 32, 200)
        self.fc2 = DdpgUtility.init_linear_layer(200, 200)
        # Hidden layers that match the number of dimensions of action
        self.action_value = DdpgUtility.init_linear_layer(
            DdpgUtility.get_bit_num(output),
            200
        )
        self.fc3 = DdpgUtility.init_linear_layer(200, 1, 0.0003)

        self.optimizer = None

    def forward(self, state, action):
        state_value = F.relu(self.conv1(state))
        state_value = F.relu(self.conv2(state_value))
        state_value = F.relu(self.conv3(state_value))
        state_value = F.relu(self.fc1(state_value.view(state_value.size(0), -1)))
        state_value = F.relu(self.fc2(state_value))

        action_value = F.relu(self.action_value(torch.unsqueeze(action, 1)))
        state_action_value = F.relu(torch.add(state_value, action_value))

        return self.fc3(state_action_value)

    def init_optimizer(self, lr):
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

class DdpgUtility(object):
    @staticmethod
    def init_conv_layer(in_channel, out_channel, kernel_size, stride):
        conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)
        # initialize the layer as supplementary information says
        f = 1 / np.sqrt(conv.weight.data.size()[0])
        return DdpgUtility._init_torchnn(conv, f)

    @staticmethod
    def init_linear_layer(in_channel, out_channel, f=None):
        fc = nn.Linear(in_channel, out_channel)
        if f == None:
            f = 1 / np.sqrt(fc.weight.data.size()[0])
        return DdpgUtility._init_torchnn(fc, f)

    @classmethod
    def _init_torchnn(cls, torchnn, value):
        torch.nn.init.uniform_(torchnn.weight.data, -value, value)
        torch.nn.init.uniform_(torchnn.bias.data, -value, value)
        return torchnn
    
    @classmethod
    def _get_bit_num(cls, number, count):
        if number == 1 or number == 0:
            return count
        
        return DdpgUtility._get_bit_num(number // 2, count + 1)

    @staticmethod
    def get_bit_num(number):
        """
        input: Decimal number
        output: Number of bits in binary number
        """
        return DdpgUtility._get_bit_num(number - 1, 1)
