import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import utility
from agent.learningmethod.dqn.network import Network
from agent.learningmethod.replaymemory import ReplayMemory


class DqnLearningMethod(object):
    def __init__(self, screen_height, screen_width, n_actions):
        self.policy_net = Network(screen_height, screen_width, n_actions).to(utility.device)
        self.target_net = Network(screen_height, screen_width, n_actions).to(utility.device)
        self.target_net.load_state_dict(self.policy_net.state_dict)
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters)
        self.memory = ReplayMemory(10000)

    def update(self):
        # optimize model
        if len(self.memory) < utility.BATCH_SIZE:
            return
        transitions = self.memory.sample(utility.BATCH_SIZE)
        batch = utility.Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=utility.device, dtype=torch.bool)
        non_final_next_state = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_value = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(utility.BATCH_SIZE, device=utility.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_state).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_value = (next_state_values * utility.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_value, expected_state_action_value.unsqueeze(1))

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
