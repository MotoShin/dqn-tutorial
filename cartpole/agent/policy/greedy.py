import torch


class Greedy(object):
    @staticmethod
    def select(set: torch.Tensor):
        return set.max(1)[1].view(1, 1)
