import unittest
import sys
import os

sys.path.append(os.path.abspath(".."))
from agent.learningmethod.generate import LearningMethodGenerate
from simulation.values.agents import AgentsNames

class TestGenerate(unittest.TestCase):
    def test_generate(self) -> None:
        expected = (
            (AgentsNames.DQN, 2, "dqn"),
            (AgentsNames.DQN_SOFTUPDATE, 2, "dqn_softupdate"),
            (AgentsNames.DQN_DUELINGNET, 2, "dqn_dueling_network"),
            (AgentsNames.DQN_SOFTUPDATE_DUELINGNET, 2, "dqn_softupdate_dueling_network"),
            (AgentsNames.DDQN, 2, "ddqn"),
            (AgentsNames.DDQN_SOFTUPDATE, 2, "ddqn_softupdate"),
            (AgentsNames.DDQN_DUELINGNET, 2, "ddqn_dueling_network"),
            (AgentsNames.DDQN_SOFTUPDATE_DUELINGNET, 2, "ddqn_softupdate_dueling_network"),
            (AgentsNames.DDPG, 2, "ddpg")
        )
        for agent_name, n_action, expected_value in expected:
            with self.subTest(agent_name=agent_name, n_action=n_action):
                lm = LearningMethodGenerate.generate(agent_name, n_action)
                self.assertEqual(lm.get_method_name(), expected_value)

if __name__ == "__main__":
    unittest.main()
