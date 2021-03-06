from abc import ABCMeta, abstractmethod
import torch.autograd as autograd

import utility


class Model(object):
    @abstractmethod
    def __init__(self, n_actions, soft_update_flg, dueling_network_flg):
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
    def save_effect(self, last_idx, action, reward, done):
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

    @abstractmethod
    def get_method_name(self):
        pass

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if utility.USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)
