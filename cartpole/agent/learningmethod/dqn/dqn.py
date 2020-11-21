import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import utility
from agent.learningmethod.dqn.network import Network
from agent.learningmethod.replaybuffer import ReplayBuffer
from agent.learningmethod.model import Model


class DqnLearningMethod(Model):
    def __init__(self, n_actions):
        self.value_net = Network(n_actions).to(utility.device)
        self.target_net = Network(n_actions).to(utility.device)
        self.target_net.load_state_dict(self.value_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.value_net.parameters(), lr=utility.NW_LEARNING_RATE, alpha=utility.NW_ALPHA, eps=utility.NW_EPS)
        self.memory = ReplayBuffer(utility.NUM_REPLAY_BUFFER, utility.FRAME_NUM)

    def optimize_model(self, target_policy):
        if not self.memory.can_sample(utility.BATCH_SIZE):
            return

        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.memory.sample(utility.BATCH_SIZE)

        obs_batch = torch.from_numpy(obs_batch).type(utility.dtype) / 255.0
        act_batch = torch.from_numpy(act_batch).long()
        rew_batch = torch.from_numpy(rew_batch)
        next_obs_batch = torch.from_numpy(next_obs_batch).type(utility.dtype) / 255.0
        not_done_mask = torch.from_numpy(1 - done_mask).type(utility.dtype)

        # Q values
        current_Q_values = self.value_net(obs_batch).gather(1, act_batch.unsqueeze(1)).squeeze(1)
        # target Q values
        next_max_q = target_policy.select(self.target_net(next_obs_batch))
        next_Q_values = not_done_mask * next_max_q
        target_Q_values = rew_batch + (utility.GAMMA * next_Q_values)
        # Compute Bellman error
        bellman_error = target_Q_values - current_Q_values
        # Clip the bellman error between [-1, 1]
        clipped_bellman_error = bellman_error.clamp(-1, 1)
        d_error = clipped_bellman_error * -1.0

        # optimize
        self.optimizer.zero_grad()
        current_Q_values.backward(d_error.data)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.value_net.state_dict())

    def save_memory(self, state):
        return self.memory.store_frame(state)

    def save_effect(self, last_idx, action, reward, done):
        self.memory.store_effect(last_idx, action, reward, done)

    def output_target_net(self, state):
        return self.target_net(state)

    def output_value_net(self, state):
        return self.value_net(state)

    def output_net_paramertes(self):
        torch.save(self.value_net.state_dict(), utility.NET_PARAMETERS_BK_PATH)

    def get_screen_history(self):
        return self.memory.encode_recent_observation()
