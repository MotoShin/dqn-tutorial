import random
import numpy as np

from domain.stepresult import StepResult

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.frame_history_len = 4
    
    def push(self, step_result: StepResult):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = step_result.get_transition()
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # TODO: 書き直し、全体的にnumpyで書き直した方が幸せな気がする
        return random.sample(self.memory, batch_size)

    def get_recent_screen(self):
        return self._get_recent_screen(self.position)

    def _get_recent_screen(self, idx):
        end_idx = idx + 1
        start_idx = end_idx - self.frame_history_len

        if start_idx < 0 and len(self.memory) != self.capacity:
            start_idx = 0
        for index in range(start_idx, end_idx - 1):
            if self.memory[index % self.capacity]:
                start_idx = index + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)

        frames = [np.zeros_like(self.memory[0].state) for _ in range(missing_context)]
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.memory[0].state) for _ in range(missing_context)]
            for index in range(start_idx, end_idx):
                frames.append(self.memory[index % self.capacity].state)
        else:
            frame_histories = StepResult.Transition(*zip(*self.memory[start_idx:end_idx]))
            for frame in frame_histories:
                frames.append(frame)

        return np.concatenate(frames, 0)
    
    def __len__(self):
        return len(self.memory)
