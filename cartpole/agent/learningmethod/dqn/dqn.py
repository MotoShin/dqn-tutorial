import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import utility
from agent.learningmethod.dqn.network import Network
from agent.learningmethod.replaymemory import ReplayMemory
from agent.learningmethod.model import Model
from domain.stepresult import StepResult


class DqnLearningMethod(Model):
    def __init__(self, n_actions):
        self.value_net = Network(n_actions).to(utility.device)
        self.target_net = Network(n_actions).to(utility.device)
        self.target_net.load_state_dict(self.value_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.value_net.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
        self.memory = ReplayMemory(10000)

    @staticmethod
    def _check_all_zeros(arr):
        np_arr = arr.numpy()
        flag = False
        for i in np_arr[0]:
            flag = flag or np.count_nonzero(i) == 0
        return flag

    def _is_not_ending(self, arr):
        return not self._check_all_zeros(arr)

    def optimize_model(self, target_policy):
        if len(self.memory) < utility.BATCH_SIZE:
            return
        transitions = self.memory.sample(utility.BATCH_SIZE)
        batch = StepResult.Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: self._is_not_ending(s), batch.next_state)), device=utility.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if self._is_not_ending(s)])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_value = self.value_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(utility.BATCH_SIZE, device=utility.device)
        next_state_values[non_final_mask] = target_policy.select(self.target_net(non_final_next_states))

        # Compute the expected Q values
        expected_state_action_value = (next_state_values * utility.GAMMA) + reward_batch

        bellman_error = state_action_value - expected_state_action_value.unsqueeze(1)
        clipped_bellman_error = bellman_error.clamp(-1, 1)
        d_error = clipped_bellman_error * -1.0

        # optimize the model
        self.optimizer.zero_grad()
        state_action_value.backward(d_error.data)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.value_net.state_dict())

    def save_memory(self, step_result):
        self.memory.push(step_result)

    def output_target_net(self, state):
        return self.target_net(state)

    def output_value_net(self, state):
        return self.value_net(state)

    def output_net_paramertes(self):
        torch.save(self.value_net.state_dict(), utility.NET_PARAMETERS_BK_PATH)
