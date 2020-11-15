from collections import deque
import numpy as np
import torch

import utility


class Input(object):
    def __init__(self):
        w = utility.RESIZE_SCREEN_SIZE_WIDTH
        h = utility.RESIZE_SCREEN_SIZE_HEIGHT
        f = utility.FRAME_NUM
        self.inputs = np.array([[np.array([0 for _ in range(w)]) for _ in range(h)] for _ in range(f)])

    def push(self, inp):
        self.inputs[0:-1] = self.inputs[1:]
        self.inputs[-1] = inp
    
    def get(self):
        return torch.from_numpy(self.inputs / 255.0).type(utility.dtype).unsqueeze(0)
