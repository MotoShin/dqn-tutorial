from simulation.values.agents import AgentsNames
from agent.agentmodel import ProbabilisticAgent, DeterministicAgent
from agent.policy.egreedy import Egreedy
from agent.policy.greedy import Greedy


class AgentGenerate(object):
    @staticmethod
    def generate(lm, env):
        if lm.get_method_name() == AgentsNames.DDPG.value:
            return DeterministicAgent(learning_method=lm, input_action_num=env.get_number_of_input_action())
        else:
            return ProbabilisticAgent(
                learning_method=lm,
                behavior_policy=Egreedy(env.get_n_actions()),
                target_policy=Greedy()
            )
