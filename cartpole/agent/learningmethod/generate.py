from agent.learningmethod.model import Model
from agent.learningmethod.dqn import DqnLearningMethod
from agent.learningmethod.ddqn import DdqnLearningMethod
from simulation.values.agents import AgentsNames

class LearningMethodGenerate(object):
    @staticmethod
    def generate(agent_name, n_actions):
        learning_method = None
        if agent_name == AgentsNames.DQN_SOFTUPDATE:
            learning_method = DqnLearningMethod(n_actions, soft_update_flg=True)
        elif agent_name == AgentsNames.DQN_DUELINGNET:
            learning_method = DqnLearningMethod(n_actions, dueling_network_flg=True)
        elif agent_name == AgentsNames.DQN_SOFTUPDATE_DUELINGNET:
            learning_method = DqnLearningMethod(n_actions, soft_update_flg=True, dueling_network_flg=True)
        elif agent_name == AgentsNames.DDQN:
            learning_method = DdqnLearningMethod(n_actions)
        elif agent_name == AgentsNames.DDQN_SOFTUPDATE:
            learning_method = DdqnLearningMethod(n_actions, soft_update_flg=True)
        elif agent_name == AgentsNames.DDQN_DUELINGNET:
            learning_method = DdqnLearningMethod(n_actions, dueling_network_flg=True)
        elif agent_name == AgentsNames.DDQN_SOFTUPDATE_DUELINGNET:
            learning_method = DdqnLearningMethod(n_actions, soft_update_flg=True, dueling_network_flg=True)
        else:
            learning_method = DqnLearningMethod(n_actions)

        return learning_method
