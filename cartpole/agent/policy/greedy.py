import torch

from agent.policy.policymodel import PolicyModel


class Greedy(PolicyModel):
    @staticmethod
    def select(self, set: torch.Tensor):
        return set.max(1)[1].view(1, 1)
