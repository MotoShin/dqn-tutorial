from abc import ABCMeta, abstractmethod


class Model(object):
    @abstractmethod
    def __init__(self, screen_height, screen_width, n_actions):
        pass

    @abstractmethod
    def optimize_model(self, target_policy):
        pass
