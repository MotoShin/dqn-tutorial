import torch
from itertools import count
import random
import time
import numpy as np

import utility
from agent.agentmodel import ProbabilisticAgent, DeterministicAgent
from agent.learningmethod.dqn import DqnLearningMethod
from agent.learningmethod.ddqn import DdqnLearningMethod
from agent.policy.greedy import Greedy
from agent.policy.egreedy import Egreedy
from agent.learningmethod.generate import LearningMethodGenerate
from environment.cartpole import CartPole
from datautil.datashaping import DataShaping
from simulation.values.agents import AgentsNames
from utility.line_notify import LineNotify


class Simulate(object):
    def __init__(self, agent_name):
        self.env = utility.TASK
        self.agent = None
        self.dulation = []
        self.episode_dulations = []
        self.reward = []
        self.episode_rewards = []
        self.agent_name = AgentsNames.value_of(agent_name)

    def agent_reset(self):
        self.env.reset()
        learning_method = LearningMethodGenerate.generate(self.agent_name, self.env.get_n_actions())
        self.agent = ProbabilisticAgent(
            learning_method=learning_method,
            behavior_policy=Egreedy(self.env.get_n_actions()),
            target_policy=Greedy()
        )

    def start(self):
        start = time.time()
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
                DataShaping.makeCsv(self.episode_dulations, 'dulation', "{}_dulation_{}.csv".format(self.agent.get_method_name(), i_simulation+1))
                DataShaping.makeCsv(self.episode_rewards, 'reward', "{}_reward_{}.csv".format(self.agent.get_method_name(), i_simulation+1))
        DataShaping.makeCsv(self.episode_dulations, 'dulation', "{}_dulation.csv".format(self.agent.get_method_name()))
        DataShaping.makeCsv(self.episode_rewards, 'reward', "{}_reward.csv".format(self.agent.get_method_name()))
        end = time.time()
        
        # Line notify
        self.env.close()
        LineNotify.send_line_notify(
            utility.LINE_NOTIFY_FLG,
            utility.LINE_NOTIFY_TOKEN,
            utility.LINE_NOTIFY_MSG.format(self._shape_time(end - start)))

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
        sim = simulate_num + 1
        epi = episode_num + 1
        late = float((simulate_num * utility.NUM_EPISODE + epi) / (utility.NUM_SIMULATION * utility.NUM_EPISODE))
        late_percent = late * 100
        if (late_percent % 10 == 0):
            print("progress: {: >3} %".format(late_percent))

    def _shape_time(self, elapsed_time):
        elapsed_time = int(elapsed_time)

        elapsed_hour = elapsed_time // 3600
        elapsed_minute = (elapsed_time % 3600) // 60
        elapsed_second = (elapsed_time % 3600 % 60)

        return str(elapsed_hour).zfill(2) + ":" \
                + str(elapsed_minute).zfill(2) + ":" \
                + str(elapsed_second).zfill(2)
