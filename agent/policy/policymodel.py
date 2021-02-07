import torch

from abc import ABCMeta, abstractmethod


class PolicyModel(object):
    @abstractmethod
    def select(self, set: torch.Tensor, epi):
        pass
