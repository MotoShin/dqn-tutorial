from collections import namedtuple


class StepResult(object):
    # TODO: doneも追加
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    def __init__(self, state, action, next_state, reward):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

    def get_transition(self):
        return self.Transition(self.state, self.action, self.next_state, self.reward)