from collections import namedtuple


class StepResult(object):
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
    def __init__(self, state, action, next_state, reward, done):
        self.transition = self.Transition(state, action, next_state, reward, done)

    def get_transition(self):
        return self.transition