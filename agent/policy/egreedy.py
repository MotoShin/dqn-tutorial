import torch
import random
import math

import utility
from utility.schedule import LinearSchedule
from agent.policy.greedy import Greedy
from agent.policy.policymodel import PolicyModel
from enum import Enum


class Egreedy(PolicyModel):
    def __init__(self, n_action):
        self.steps_done = 0
        self.n_action = n_action
        self.exploration_schedule = LinearSchedule(utility.EPS_TIMESTEPS, utility.EPS_END, utility.EPS_START)

    def select(self, set: torch.Tensor, epi):
        sample = random.random()
        selected = None

        value = None
        if EgreedyOptions.EPISODE == utility.EPS_MODE:
            value = self.exploration_schedule.value(epi)
        elif EgreedyOptions.ACTION == utility.EPS_MODE:
            value = self.exploration_schedule.value(self.steps_done)
        else:
            raise ValueError("EPS_MODE is undefined value.")

        if sample > value:
            selected =  Greedy().select(set)
        else:
            selected = random.randrange(self.n_action)
        self.steps_done += 1
        return selected

class EgreedyOptions(Enum):
    EPISODE = "episode"
    ACTION = "action"
