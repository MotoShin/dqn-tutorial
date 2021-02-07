import torch

from enum import Enum


class GreedyEnum(Enum):
    DEFAULT = "default"
    INDEX = "index"

class Greedy(object):
    @staticmethod
    def select(set: torch.Tensor, mode=GreedyEnum.DEFAULT):
        if len(set) == 1:
            # select action
            return set.max(1)[1].view(1, 1).item()

        # replay buffer
        if mode == GreedyEnum.INDEX:
            """
            index array
            tensor([[0], [0], ...])
            """
            return torch.max(set, 1)[1].unsqueeze(1)
        else:
            """
            value array
            tensor([[0.001], [0.002], ...])
            """
            return set.detach().max(1)[0]
