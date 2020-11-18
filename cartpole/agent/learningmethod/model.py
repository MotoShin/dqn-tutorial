from abc import ABCMeta, abstractmethod


class Model(object):
    @abstractmethod
    def __init__(self, n_actions):
        pass

    @abstractmethod
    def optimize_model(self, target_policy):
        pass

    @abstractmethod
    def update_target_network(self):
        pass

    @abstractmethod
    def save_memory(self, step_result):
        pass

    @abstractmethod
    def output_value_net(self, state):
        pass

    @abstractmethod
    def output_target_net(self, state):
        pass

    @abstractmethod
    def output_net_paramertes(self):
        pass

    @abstractmethod
    def get_screen_history(self):
        pass
