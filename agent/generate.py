from simulation.values.agents import AgentsNames
from agent.agentmodel import ProbabilisticAgent, DeterministicAgent
from agent.policy.egreedy import Egreedy


class AgentGenerate(object):
    @staticmethod
    def generate(lm, env):
        if lm.get_method_name() == AgentsNames.DDPG.value:
            return DeterministicAgent(learning_method=lm, env)
        else:
            return ProbabilisticAgent(
                learning_method=lm,
                behavior_policy=Egreedy(env.get_n_actions()),
                target_policy=Greedy()
            )
