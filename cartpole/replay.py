import torch
from itertools import count

import utility
from environment.cartpole import CartPole
from agent.learningmethod.dqn.network import Network
from agent.policy.greedy import Greedy
from domain.networkinput import Input


if __name__ == '__main__':
    env = CartPole()
    env.reset()
    net = Network(env.get_n_actions())
    net.load_state_dict(torch.load(utility.NET_PARAMETERS_BK_PATH))
    policy = Greedy()

    screen = env.get_screen()
    state = Input()
    state.push(screen)
    next_state = Input()
    next_state.push(screen)
    for i in count():
        action = policy.select(net(state.get()))
        _, _, done, _ = env.step(action.item())

        screen = env.get_screen()
        if not done:
            next_state.push(screen)
        else:
            env.reset()
            next_state.push(env.get_screen())
        state.push(screen)
        if done:
            print("step: {}".format(i))
            break
