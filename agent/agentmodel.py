import torch
import numpy as np

import utility
from agent.learningmethod.model import Model
from agent.policy.policymodel import PolicyModel
from agent.learningmethod.noise.noiseinjectory import OrnsteinUhlenbeckActionNoise


class ProbabilisticAgent(object):
    def __init__(self, learning_method: Model, behavior_policy: PolicyModel, target_policy: PolicyModel):
        self.learning_method = learning_method
        # 行動方策
        self.behavior_policy = behavior_policy
        # 推定方策
        self.target_policy = target_policy

    def select_action(self, state, epi):
        return self.behavior_policy.select(self.learning_method.output_value_net(state), epi)

    def update(self):
        self.learning_method.optimize_model(self.target_policy)

    def update_target_network(self):
        self.learning_method.update_target_network()

    def save_memory(self, step_result):
        return self.learning_method.save_memory(step_result)

    def save_effect(self, last_idx, action, reward, done):
        self.learning_method.save_effect(last_idx, action, reward, done)
    
    def get_screen_history(self):
        return self.learning_method.get_screen_history()

    def save_parameters(self):
        self.learning_method.output_net_paramertes()

    def get_method_name(self):
        return self.learning_method.get_method_name()

class DeterministicAgent(object):
    def __init__(self, learning_method: Model, input_action_num: int):
        self.learning_method = learning_method
        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(input_action_num))
        # TODO: 引数で設定できるようにする
        self.action_minimum = 0.0
        self.action_maximum = 1.0

    def select_action(self, state, epi):
        mu = self.learning_method.output_value_net(state)
        mu_w_noise = mu + torch.tensor(self.noise(), dtype=torch.float).to(device=utility.device)
        action = self._change_range(np.clip(mu_w_noise.view(1).item(), -1, 1))
        action = self._round_action_number(action)
        # print(action)
        return action

    def update(self):
        self.learning_method.optimize_model()

    def update_target_network(self):
        self.learning_method.update_target_network()

    def save_memory(self, step_result):
        return self.learning_method.save_memory(step_result)

    def save_effect(self, last_idx, action, reward, done):
        self.learning_method.save_effect(last_idx, action, reward, done)

    def get_screen_history(self):
        return self.learning_method.get_screen_history()

    def save_parameters(self):
        self.learning_method.output_net_paramertes()

    def get_method_name(self):
        return self.learning_method.get_method_name()

    def _change_range(self, action):
        network_output_minimum = -1.0
        network_output_maximum = 1.0

        rate = (network_output_maximum - action) / (network_output_maximum + abs(network_output_minimum))
        changed_range_action = (self.action_maximum - self.action_minimum) * rate + self.action_minimum
        return changed_range_action

    def _round_action_number(self, action):
        reference_value = float((self.action_minimum + self.action_maximum) / 2)
        if abs(1.0 - action) >= reference_value:
            return 0
        else:
            return 1
