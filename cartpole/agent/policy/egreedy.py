import torch
import random
import math

import utility
from agent.policy.greedy import Greedy
from agent.policy.policymodel import PolicyModel


class Egreedy(PolicyModel):
    def __init__(self, n_action):
        self.epsilon_start = utility.EPS_START
        self.epsilon_end = utility.EPS_END
        self.epsilon_decay = utility.EPS_DECAY
        self.steps_done = 0
        self.n_action = n_action

    def select(self, set: torch.Tensor):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1 * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if sample < eps_threshold:
            return Greedy().select(set)
        else:
            return torch.tensor([[random.randrange(self.n_action)]], device=utility.device, dtype=torch.long)
