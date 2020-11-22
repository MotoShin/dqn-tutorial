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
        return self.behavior_policy.select(self.learning_method.output_value_net(state))

    def update(self):
        self.learning_method.optimize_model(self.target_policy)

    def update_target_network(self):
        self.learning_method.update_target_network()

    def save_memory(self, step_result):
        return self.learning_method.save_memory(step_result)

    def save_effect(self, last_idx, action, reward, done):
        self.learning_method.save_effect(last_idx, action, reward, done)
    
    def get_screen_history(self):
        return self.learning_method.get_screen_history()

    def save_parameters(self):
        self.learning_method.output_net_paramertes()

    def get_method_name(self):
        return self.learning_method.get_method_name()
