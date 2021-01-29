import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

import utility
from agent.learningmethod.network.dqnnetwork import DqnNetwork
from agent.learningmethod.network.duelingnetwork import DqnDuelingNetwork
from agent.learningmethod.replaybuffer import ReplayBuffer
from agent.learningmethod.model import Model, Variable
from simulation.values.agents import AgentsNames


class DqnLearningMethod(Model):
    def __init__(self, n_actions, soft_update_flg=False, dueling_network_flg=False):
        self.soft_update_flg = soft_update_flg
        if dueling_network_flg:
            self.value_net = DqnDuelingNetwork(n_actions).type(utility.dtype).to(device=utility.device)
            self.target_net = DqnDuelingNetwork(n_actions).type(utility.dtype)   
        else:
            self.value_net = DqnNetwork(n_actions).type(utility.dtype).to(device=utility.device)
            self.target_net = DqnNetwork(n_actions).type(utility.dtype)   
        self.target_net.load_state_dict(self.value_net.state_dict())
        self.target_net.eval()
        self.target_net.to(device=utility.device)
        self.optimizer = optim.RMSprop(self.value_net.parameters(), lr=utility.NW_LEARNING_RATE, alpha=utility.NW_ALPHA, eps=utility.NW_EPS)
        self.memory = ReplayBuffer(utility.NUM_REPLAY_BUFFER, utility.FRAME_NUM)

    def optimize_model(self, target_policy):
        if not self.memory.can_sample(utility.BATCH_SIZE):
            return

        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.memory.sample(utility.BATCH_SIZE)

        obs_batch = Variable(torch.from_numpy(obs_batch).type(utility.dtype) / 255.0)
        act_batch = Variable(torch.from_numpy(act_batch).long())
        rew_batch = Variable(torch.from_numpy(rew_batch))
        next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(utility.dtype) / 255.0)
        not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(utility.dtype)

        if utility.USE_CUDA:
            act_batch = act_batch.cuda()
            rew_batch = rew_batch.cuda()

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

        # target network soft update
        if self.soft_update_flg:
            self._soft_update_target_network()

    def update_target_network(self):
        if self.soft_update_flg:
            pass
        else:
            self.target_net.load_state_dict(self.value_net.state_dict())

    def _soft_update_target_network(self):
        for target_param, value_param in zip(self.target_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(utility.TAU * value_param.data + (1.0 - utility.TAU) * target_param.data)

    def save_memory(self, state):
        return self.memory.store_frame(state)

    def save_effect(self, last_idx, action, reward, done):
        self.memory.store_effect(last_idx, action, reward, done)

    def output_target_net(self, state):
        output = None
        with torch.no_grad():
            state = Variable(state)
            state.to(utility.device)
            output = self.target_net(state)
        return output

    def output_value_net(self, state):
        output = None
        with torch.no_grad():
            state = Variable(state)
            state.to(utility.device)
            output = self.value_net(state)
        return output

    def output_net_paramertes(self):
        torch.save(self.value_net.state_dict(), utility.NET_PARAMETERS_BK_PATH)

    def get_screen_history(self):
        return self.memory.encode_recent_observation()

    def get_method_name(self):
        return AgentsNames.DQN.value
