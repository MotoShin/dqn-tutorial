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

    def select_action(self, state, epi):
        mu = self.learning_method.output_value_net(state)
        mu_w_noise = mu + torch.tensor(self.noise(), dtype=torch.float).to(device=utility.device)
        action = mu_w_noise.view(1).item()
        return int(np.clip(action, 0, 1))

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
