from abc import ABCMeta, abstractmethod


class Model(object):
    @abstractmethod
    def __init__(self, screen_height, screen_width, n_actions):
        pass

    @abstractmethod
    def optimize_model(self, target_policy):
        pass

    @abstractmethod
    def update_target_network(self):
        pass

    @abstractmethod
    def save_memory(self, state, action, next_state, reward):
        pass

    @abstractmethod
    def output_policy_net(self, state):
        pass

    @abstractmethod
    def output_target_net(self, state):
        pass
