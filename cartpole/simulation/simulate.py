import torch
from itertools import count
import random
import numpy as np

import utility
from agent.agentmodel import Agent
from agent.learningmethod.dqn.dqn import DqnLearningMethod
from agent.learningmethod.dqn.dqnsoftupdate import DqnSoftUpdateLearningMethod
from agent.policy.greedy import Greedy
from agent.policy.egreedy import Egreedy
from environment.cartpole import CartPole
from datautil.datashaping import DataShaping
from simulation.values.agnets import AgentsNames


class Simulate(object):
    def __init__(self, agent_name):
        self.env = CartPole()
        self.agent = None
        self.dulation = []
        self.episode_dulations = []
        self.reward = []
        self.episode_rewards = []
        self.agent_name = AgentsNames.value_of(agent_name)

    def agent_reset(self):
        self.env.reset()
        if self.agent_name == AgentsNames.DQN_SOFTUPFATE:
            learning_method = DqnSoftUpdateLearningMethod(self.env.get_n_actions())
        else:
            learning_method = DqnLearningMethod(self.env.get_n_actions())
        self.agent = Agent(
            learning_method=learning_method,
            behavior_policy=Egreedy(self.env.get_n_actions()),
            target_policy=Greedy()
        )

    def start(self):
        print('Start!')
        for i_simulation in range(utility.NUM_SIMULATION):
            self.agent_reset()
            self.dulation = []
            self.reward = []
            self.one_simulate_start(i_simulation)
            self.agent.save_parameters()
            self.episode_dulations.append(self.dulation)
            self.episode_rewards.append(self.reward)
            if (i_simulation + 1) % 10 == 0:
                print(i_simulation + 1)
                DataShaping.makeCsv(self.episode_dulations, ['episode', 'dulation'], "{}_dulation_{}.csv".format(self.agent.get_method_name, i_simulation+1))
                DataShaping.makeCsv(self.episode_rewards, ['episode', 'reward'], "{}_reward_{}.csv".format(self.agent.get_method_name(), i_simulation+1))
        DataShaping.makeCsv(self.episode_dulations, ['episode', 'dulation'], "{}_dulation.csv".format(self.agent.get_method_name()))
        DataShaping.makeCsv(self.episode_rewards, ['episode', 'reward'], "{}_reward.csv".format(self.agent.get_method_name()))
        self.env.close()
        print('Complete')

    def one_simulate_start(self, simulation_num):
        for i_episode in range(utility.NUM_EPISODE):
            self.env.reset()
            self.one_episode_start(simulation_num, i_episode)
            if i_episode % utility.TARGET_UPDATE == 0:
                self.agent.update_target_network()

    def one_episode_start(self, simulation_num, episode_num):
        self.output_progress(simulation_num, episode_num)
        current_screen = self.env.get_screen()
        state = current_screen
        sum_reward = 0.0
        for t in count():
            # Store frame
            last_idx = self.agent.save_memory(state)

            # Chose action
            recent_screen = self.agent.get_screen_history()
            inp = torch.from_numpy(np.array([recent_screen])).type(utility.dtype) / 255.0
            action = self.agent.select_action(inp, episode_num)
            # Action
            _, reward, done, _ = self.env.step(action)

            screen = self.env.get_screen()
            if done:
                reward = self.env.episode_end_reward(reward)
                self.env.reset()
                screen = self.env.get_screen()
            # Move to the next state
            sum_reward += reward
            state = screen

            # Store the effect in memory
            self.agent.save_effect(last_idx, action, reward, done)
            
            self.agent.update()
            if done:
                self.dulation.append(t + 1)
                self.reward.append(sum_reward)
                break

    def output_progress(self, simulate_num, episode_num):
        late = float((simulate_num * utility.NUM_SIMULATION + episode_num) / (utility.NUM_SIMULATION * utility.NUM_EPISODE))
        late_percent = late * 100
        if (late_percent % 10 == 0):
            print("progress: {: >3} %".format(late_percent))
