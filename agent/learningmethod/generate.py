from agent.learningmethod.model import Model
from agent.learningmethod.dqn import DqnLearningMethod
from agent.learningmethod.ddqn import DdqnLearningMethod
from agent.learningmethod.ddpg import DdpgLearningMethod
from simulation.values.agents import AgentsNames

class LearningMethodGenerate(object):
    @staticmethod
    def generate(agent_name, n_actions):
        learning_method = None
        if agent_name.value == AgentsNames.DQN_SOFTUPDATE.value:
            learning_method = DqnLearningMethod(n_actions, soft_update_flg=True)
        elif agent_name.value == AgentsNames.DQN_DUELINGNET.value:
            learning_method = DqnLearningMethod(n_actions, dueling_network_flg=True)
        elif agent_name.value == AgentsNames.DQN_SOFTUPDATE_DUELINGNET.value:
            learning_method = DqnLearningMethod(n_actions, soft_update_flg=True, dueling_network_flg=True)
        elif agent_name.value == AgentsNames.DDQN.value:
            learning_method = DdqnLearningMethod(n_actions)
        elif agent_name.value == AgentsNames.DDQN_SOFTUPDATE.value:
            learning_method = DdqnLearningMethod(n_actions, soft_update_flg=True)
        elif agent_name.value == AgentsNames.DDQN_DUELINGNET.value:
            learning_method = DdqnLearningMethod(n_actions, dueling_network_flg=True)
        elif agent_name.value == AgentsNames.DDQN_SOFTUPDATE_DUELINGNET.value:
            learning_method = DdqnLearningMethod(n_actions, soft_update_flg=True, dueling_network_flg=True)
        elif agent_name.value == AgentsNames.DDPG.value:
            learning_method = DdpgLearningMethod(n_actions)
        else:
            learning_method = DqnLearningMethod(n_actions)

        return learning_method
