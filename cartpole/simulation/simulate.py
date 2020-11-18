import torch
from itertools import count
import random

import utility
from agent.agentmodel import Agent
from agent.learningmethod.dqn.dqn import DqnLearningMethod
from agent.policy.greedy import Greedy
from agent.policy.egreedy import Egreedy
from environment.cartpole import CartPole
from datautil.datashaping import DataShaping
from domain.networkinput import Input
from domain.stepresult import StepResult


class Simulate(object):
    def __init__(self):
        self.env = CartPole()
        self.agent = None
        self.dulation = []
        self.episode_dulations = []
        self.reward = []
        self.episode_rewards = []

    def agent_reset(self):
        self.env.reset()
        self.agent = Agent(
            learning_method=DqnLearningMethod(self.env.get_n_actions()),
            behavior_policy=Egreedy(self.env.get_n_actions()),
            target_policy=Greedy()
        )

    def start(self):
        for i_simulation in range(utility.NUM_SIMULATION):
            self.agent_reset()
            self.dulation = []
            self.reward = []
            self.one_simulate_start()
            self.agent.save_parameters()
            self.episode_dulations.append(self.dulation)
            self.episode_rewards.append(self.reward)
            if (i_simulation + 1) % 10 == 0:
                print(i_simulation + 1)
                DataShaping.makeCsv(self.episode_dulations, ['episode', 'dulation'], "dulation_{}.csv".format(i_simulation+1))
                DataShaping.makeCsv(self.episode_rewards, ['episode', 'reward'], "reward_{}.csv".format(i_simulation+1))
        DataShaping.makeCsv(self.episode_dulations, ['episode', 'dulation'], 'dulation.csv')
        DataShaping.makeCsv(self.episode_rewards, ['episode', 'reward'], 'reward.csv')
        print('Complete')

    def one_simulate_start(self):
        for i_episode in range(utility.NUM_EPISODE):
            self.env.reset()
            self.one_episode_start()
            if i_episode % utility.TARGET_UPDATE == 0:
                self.agent.update_target_network()

    def one_episode_start(self):
        current_screen = self.env.get_screen()
        state = current_screen
        next_state = current_screen
        sum_reward = 0.0
        for t in count():
            if t > utility.BATCH_SIZE:
                action = self.agent.select_action(self.agent.get_screen_history())
            else:
                action = torch.tensor([[random.randrange(self.env.get_n_actions())]])
            _, reward, done, _ = self.env.step(action.item())

            screen = self.env.get_screen()
            if done:
                reward = self.env.episode_end_reward(reward)
                self.env.reset()
                screen = self.env.get_screen()
            next_state = screen
            reward = torch.tensor([reward], device=utility.device)

            # Store the transaction in memory
            step_result = StepResult(state, action, next_state, reward, done)
            self.agent.save_memory(step_result)
            
            # Move to the next state
            state = next_state
            sum_reward += reward.item()

            self.agent.update()
            if done:
                self.dulation.append(t + 1)
                self.reward.append(sum_reward)
                break
