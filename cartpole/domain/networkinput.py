from collections import deque
import numpy as np
import torch

import utility


class Input(object):
    def __init__(self, init_screen):
        w = utility.RESIZE_SCREEN_SIZE_WIDTH
        h = utility.RESIZE_SCREEN_SIZE_HEIGHT
        f = utility.FRAME_NUM
        self.inputs = np.array([init_screen for _ in range(f)])

    def push(self, inp):
        self.inputs[0:-1] = self.inputs[1:]
        self.inputs[-1] = inp
    
    def zero_push(self):
        w = utility.RESIZE_SCREEN_SIZE_WIDTH
        h = utility.RESIZE_SCREEN_SIZE_HEIGHT
        self.push([np.array([0 for _ in range(w)]) for _ in range(h)])

    def get(self):
        return torch.from_numpy(self.inputs / 255.0).type(utility.dtype).unsqueeze(0)
