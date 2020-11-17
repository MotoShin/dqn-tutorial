import torch
import random
import math

import utility
from agent.policy.greedy import Greedy
from agent.policy.policymodel import PolicyModel


class Egreedy(PolicyModel):
    def __init__(self, n_action):
        self.epsilon_threshold = utility.EPS_START
        self.epsilon_end = utility.EPS_END
        self.epsilon_decay = utility.EPS_DECAY
        self.steps_done = 0
        self.n_action = n_action

    def select(self, set: torch.Tensor):
        sample = random.random()
        self.steps_done += 1
        if sample < self.epsilon_threshold:
            return torch.tensor([[random.randrange(self.n_action)]], device=utility.device, dtype=torch.long)
        else:
            return Greedy().select(set)
        self.epsilon_threshold *= self.epsilon_decay
        self.epsilon_decay = max(self.epsilon_end, self.epsilon_threshold)
