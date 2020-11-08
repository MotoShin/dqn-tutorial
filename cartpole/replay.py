import torch
from itertools import count

import utility
from environment.cartpole import CartPole
from agent.learningmethod.dqn.network import Network
from agent.policy.greedy import Greedy


if __name__ == '__main__':
    env = CartPole()
    env.reset()
    _, _, screen_height, screen_width = env.get_screen().shape
    net = Network(screen_height, screen_width, env.get_n_actions())
    net.load_state_dict(torch.load(utility.NET_PARAMETERS_BK_PATH))
    policy = Greedy()

    last_screen = env.get_screen()
    current_screen = env.get_screen()
    state = current_screen = last_screen
    for i in count():
        action = policy.select(net(state))
        _, _, done, _ = env.step(action.item())

        last_screen = current_screen
        current_screen = env.get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None
        state = next_state
        if done:
            print("step: {}".format(i))
            break
