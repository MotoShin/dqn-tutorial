import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import utility
from agent.learningmethod.dqn.network import Network
from agent.learningmethod.replaymemory import ReplayMemory


class DqnLearningMethod(object):
    def __init__(self, screen_height, screen_width, n_actions):
        self.policy_net = Network(screen_height, screen_width, n_actions).to(utility.device)
        self.target_net = Network(screen_height, screen_width, n_actions).to(utility.device)
        self.target_net.load_state_dict(self.policy_net.state_dict)
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters)
        self.memory = ReplayMemory(10000)

    def update(self):
        # optimize model
        hogehoge
