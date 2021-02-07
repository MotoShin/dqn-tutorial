import torch
from itertools import count
import numpy as np

import utility
from environment.cartpole import CartPole
from agent.learningmethod.dqn.network import Network
from agent.learningmethod.dqn.dqn import Variable
from agent.policy.greedy import Greedy
from agent.learningmethod.replaybuffer import ReplayBuffer


if __name__ == '__main__':
    env = CartPole()
    env.reset()
    net = Network(env.get_n_actions()).type(utility.dtype)
    net.load_state_dict(torch.load(utility.NET_PARAMETERS_BK_PATH))
    policy = Greedy()
    memory = ReplayBuffer(utility.NUM_REPLAY_BUFFER, utility.FRAME_NUM)

    screen = env.get_screen()
    state = screen
    for i in count():
        memory.store_frame(state)
        inp = torch.from_numpy(np.array([memory.encode_recent_observation()])).type(utility.dtype) / 255.0
        with torch.no_grad():
            inp = Variable(inp)
        action = policy.select(net(inp))
        _, reward, done, _ = env.step(action)
        print("done: {}, reward: {}".format(done, reward))

        screen = env.get_screen()
        if done:
            env.reset()
            screen = env.get_screen()
        state = screen
        if done:
            print("step: {}".format(i))
            break
