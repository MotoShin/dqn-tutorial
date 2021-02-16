import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

import utility
from agent.learningmethod.network.ddpgnetwork import ActorNetwork, CriticNetwork
from agent.learningmethod.replaybuffer import ReplayBuffer
from agent.learningmethod.model import Model, Variable
from simulation.values.agents import AgentsNames


class DdpgLearningMethod(Model):
    def __init__(self, n_actions, soft_update_flg=False, dueling_network_flg=False):
        self.actor = ActorNetwork(output=1).type(utility.dtype).to(device=utility.device)
        self.target_actor = ActorNetwork(output=1).type(utility.dtype)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor.eval()
        self.target_actor.to(device=utility.device)
        # init optimizer
        self.actor.init_optimizer(utility.ACTOR_LEARNING_RATE)
        self.target_actor.init_optimizer(utility.ACTOR_LEARNING_RATE)

        self.critic = CriticNetwork(n_actions).type(utility.dtype).to(device=utility.device)
        self.target_critic = CriticNetwork(n_actions).type(utility.dtype)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()
        self.target_critic.to(device=utility.device)
        # init optimizer
        self.critic.init_optimizer(utility.CRITIC_LEARNING_RATE)
        self.target_critic.init_optimizer(utility.CRITIC_LEARNING_RATE)

        self.memory = ReplayBuffer(utility.DDPG_NUM_REPLAY_BUFFER, utility.FRAME_NUM)

    def optimize_model(self, target_policy=None):
        BATCH_SIZE = utility.DDPG_BATCH_SIZE
        if not self.memory.can_sample(BATCH_SIZE):
            return

        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.memory.sample(BATCH_SIZE)

        obs_batch = Variable(torch.from_numpy(obs_batch).type(utility.dtype) / 255.0)
        act_batch = Variable(torch.from_numpy(act_batch).float())
        rew_batch = Variable(torch.from_numpy(rew_batch))
        next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(utility.dtype) / 255.0)
        not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(utility.dtype)

        if utility.USE_CUDA:
            act_batch = act_batch.cuda()
            rew_batch = rew_batch.cude()

        # Q values
        current_Q_values = self.critic(obs_batch, act_batch)
        # target Q values
        next_actions = self.target_actor(next_obs_batch)
        next_Q_values = self.target_critic(next_obs_batch, next_actions.squeeze(1))
        target_Q_values = rew_batch.unsqueeze(1) + (utility.DDPG_GAMMA * next_Q_values)
        # Compute Critic Error
        critic_error = F.mse_loss(target_Q_values, current_Q_values)

        # critic optimize
        self.critic.optimizer.zero_grad()
        critic_error.backward()
        self.critic.optimizer.step()

        # actor optimize
        self.actor.optimizer.zero_grad()
        mu = self.actor(obs_batch)
        actor_q = self.critic(obs_batch, mu)
        actor_loss = torch.mean(-actor_q)
        actor_loss.backward()
        self.actor.optimizer.step()

        # soft update is default
        self._soft_update_target_network()

    def update_target_network(self):
        pass

    def _soft_update_target_network(self):
        for target_param, value_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(utility.TAU * value_param.data + (1.0 - utility.TAU) * target_param.data)

        for target_param, value_param in zip(self.target_critic.parameters(), self.critic.parameters()):
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
            output = self.target_actor(state)
        return output

    def output_value_net(self, state):
        output = None
        with torch.no_grad():
            state = Variable(state)
            state.to(utility.device)
            output = self.actor(state)
        return output

    def output_net_paramertes(self):
        torch.save(self.actor.state_dict(), utility.NET_PARAMETERS_BK_PATH_ACTOR)
        torch.save(self.critic.state_dict(), utility.NET_PARAMETERS_BK_PATH_CRITIC)

    def get_screen_history(self):
        return self.memory.encode_recent_observation()

    def get_method_name(self):
        return AgentsNames.get_name("ddpg", False, False)
