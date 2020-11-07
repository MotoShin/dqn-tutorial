import torch
from itertools import count

import utility
from agent.agentmodel import Agent
from agent.learningmethod.dqn.dqn import DqnLearningMethod
from agent.policy.greedy import Greedy
from agent.policy.egreedy import Egreedy
from environment.cartpole import CartPole
from datautil.datashaping import DataShaping


class Simulate(object):
    def __init__(self):
        self.env = CartPole()
        self.agent = None
        self.dulation = []
        self.episode_dulations = []
        # self.reward = []
        # self.episode_rewards = []

    def agent_reset(self):
        self.env.reset()
        _, _, screen_height, screen_width = self.env.get_screen().shape
        self.agent = Agent(
            learning_method=DqnLearningMethod(screen_height, screen_width, self.env.get_n_actions()),
            behavior_policy=Egreedy(self.env.get_n_actions()),
            target_policy=Greedy()
        )

    def start(self):
        for i_simulation in range(utility.NUM_SIMULATION):
            self.agent_reset()
            self.dulation = []
            self.one_simulate_start()
            self.episode_dulations.append(self.dulation)
            if (i_simulation + 1) % 10 == 0:
                print(i_simulation + 1)
                DataShaping.makeCsv(self.episode_dulations, ['episode', 'dulation'], "dulation_{}.csv".format(i_simulation+1))
        DataShaping.makeCsv(self.episode_dulations, ['episode', 'dulation'], 'dulation.csv')
        print('Complete')

    def one_simulate_start(self):
        for i_episode in range(utility.NUM_EPISODE):
            self.env.reset()
            self.one_episode_start()
            if i_episode % utility.TARGET_UPDATE == 0:
                self.agent.update_target_network()

    def one_episode_start(self):
        last_screen = self.env.get_screen()
        current_screen = self.env.get_screen()
        state = current_screen - last_screen
        for t in count():
            action = self.agent.select_action(state)
            _, reward, done, _ = self.env.step(action.item())
            reward = torch.tensor([reward], device=utility.device)

            last_screen = current_screen
            current_screen = self.env.get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transaction in memory
            self.agent.save_memory(state, action, next_state, reward)
            
            # Move to the next state
            state = next_state

            self.agent.update()
            if done:
                self.dulation.append(t + 1)
                break
