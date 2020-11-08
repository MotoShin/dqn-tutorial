import torch

from agent.policy.policymodel import PolicyModel


class Greedy(PolicyModel):
    def select(self, set: torch.Tensor):
        if len(set) == 1:
            # select action
            return set.max(1)[1].view(1, 1)
        else:
            # replay buffer
            return set.max(1)[0].detach()
