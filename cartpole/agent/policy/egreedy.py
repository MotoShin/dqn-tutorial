import torch
import random
import math

import utility
from utility.schedule import LinearSchedule
from agent.policy.greedy import Greedy
from agent.policy.policymodel import PolicyModel


class Egreedy(PolicyModel):
    def __init__(self, n_action):
        self.steps_done = 0
        self.n_action = n_action
        self.exploration_schedule = LinearSchedule(utility.EPS_TIMESTEPS, utility.EPS_END, utility.EPS_START)

    def select(self, set: torch.Tensor):
        sample = random.random()
        selected = None
        if sample > self.exploration_schedule.value(self.steps_done):
            selected =  Greedy().select(set)
        else:
            selected = random.randrange(self.n_action)
        self.steps_done += 1
        return selected
