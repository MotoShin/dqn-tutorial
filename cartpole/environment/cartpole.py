import gym
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

import utility


class CartPole(object):
    resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

    def __init__(self):
        self.env = gym.make('CartPole-v1').unwrapped

    def get_screen(self):
        screen = self.env.render('rgb_array')
        return utility.resize_and_grayscale(screen)[0]
    
    def get_n_actions(self):
        return self.env.action_space.n
    
    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)
