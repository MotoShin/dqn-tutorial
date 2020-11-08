from agent.learningmethod.model import Model
from agent.policy.policymodel import PolicyModel


class Agent(object):
    def __init__(self, learning_method: Model, behavior_policy: PolicyModel, target_policy: PolicyModel):
        self.learning_method = learning_method
        # 行動方策
        self.behavior_policy = behavior_policy
        # 推定方策
        self.target_policy = target_policy

    def select_action(self, state):
        return self.behavior_policy.select(self.learning_method.output_policy_net(state))

    def update(self):
        self.learning_method.optimize_model(self.target_policy)

    def update_target_network(self):
        self.learning_method.update_target_network()

    def save_memory(self, state, action, next_state, reward):
        self.learning_method.save_memory(state, action, next_state, reward)

    def save_parameters(self):
        self.learning_method.output_net_paramertes()
