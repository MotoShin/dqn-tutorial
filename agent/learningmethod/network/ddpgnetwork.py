import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, output):
        super(ActorNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.fc3 = nn.Linear(18 * 18 * 32, 400)
        self.fc4 = nn.Linear(400, 300)
        self.fc5 = nn.Linear(300, output)

        self.optimizer = None

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc3(x.view(x.size(0), -1)))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return torch.tanh(x)

    def init_optimizer(self, lr):
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

class CriticNetwork(nn.Module):
    def __init__(self, output):
        super(CriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)

        self.action_fc1 = nn.Linear(DdpgUtility.get_bit_num(output), 64)
        self.action_fc2 = nn.Linear(64, 64)
        self.fc1 = nn.Linear(18 * 18 * 32 + 64, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

        self.optimizer = None

    def forward(self, state, action):
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = state.view(state.size(0), -1)

        action_latent = F.relu(self.action_fc1(action.unsqueeze(1)))
        action_latent = F.relu(self.action_fc2(action_latent))

        x = torch.cat([state, action_latent], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

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
