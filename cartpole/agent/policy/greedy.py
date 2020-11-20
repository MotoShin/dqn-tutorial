import torch


class Greedy(object):
    @staticmethod
    def select(set: torch.Tensor):
        if len(set) == 1:
            # select action
            return set.max(1)[1].view(1, 1).item()
        else:
            # replay buffer
            return set.detach().max(1)[0]
