import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, output):
        super(ActorNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 32, 200)
        self.fc5 = nn.Linear(200, output)

        self.optimizer = None

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        x = self.fc5(x)
        return torch.tanh(x)

    def init_optimizer(self, lr):
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

class CriticNetwork(nn.Module):
    def __init__(self, output):
        super(CriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1)

        self.action_fc1 = nn.Linear(DdpgUtility.get_bit_num(output), 64)
        self.action_fc2 = nn.Linear(64, 64)
        self.fc1 = nn.Linear(7 * 7 * 32 + 64, 200)
        self.fc2 = nn.Linear(200, 1)

        self.optimizer = None

    def forward(self, state, action):
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))
        state = state.view(state.size(0), -1)

        action_latent = F.relu(self.action_fc1(action.unsqueeze(1)))
        action_latent = F.relu(self.action_fc2(action_latent))

        x = torch.cat([state, action_latent], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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

class ActorOutputControl(object):
    def __init__(self):
        # TODO: 初期化で変えられるようにする
        self.output_maximum = 1.0
        self.output_minimum = -1.0

        self.action_maximum = 1.0
        self.action_minimum = 0.0

    def change_output_range(self, numOrTensor):
        rate = (self.output_maximum - numOrTensor) / (self.output_maximum + abs(self.output_minimum))
        changed_range_action = (self.action_maximum - self.action_minimum) * rate + self.action_minimum
        return changed_range_action

    def num_to_round_action_number(self, num: float):
        reference_value = float((self.action_minimum + self.action_maximum) / 2)
        if abs(1.0 - num) >= reference_value:
            return int(self.action_minimum)
        else:
            return int(self.action_maximum)

    def tensor_to_round_action_number(self, tensor: torch.tensor):
        reference_value = float((self.action_minimum + self.action_maximum) / 2)
        ary = tensor.detach().clone().numpy()
        for index in range(len(ary)):
            ary[index] = np.array(self.num_to_round_action_number(ary[index][0]))
        return torch.from_numpy(ary)
