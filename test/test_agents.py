import unittest
import sys
import os

sys.path.append(os.path.abspath(".."))
from simulation.values.agents import AgentsNames

class TestAgentsNames(unittest.TestCase):
    def test_get_name(self) -> None:
        expected = (
            ("dqn", False, False, AgentsNames.DQN.value),
            ("dqn", True, False, AgentsNames.DQN_SOFTUPDATE.value),
            ("dqn", False, True, AgentsNames.DQN_DUELINGNET.value),
            ("dqn", True, True, AgentsNames.DQN_SOFTUPDATE_DUELINGNET.value),
            ("ddqn", False, False, AgentsNames.DDQN.value),
            ("ddqn", True, False, AgentsNames.DDQN_SOFTUPDATE.value),
            ("ddqn", False, True, AgentsNames.DDQN_DUELINGNET.value),
            ("ddqn", True, True, AgentsNames.DDQN_SOFTUPDATE_DUELINGNET.value),
            ("ddpg", False, False, AgentsNames.DDPG.value)
        )
        for category, soft_update_flg, dueling_network_flg, expected_value in expected:
            with self.subTest(
                category=category,
                soft_update_flg=soft_update_flg,
                dueling_network_flg=dueling_network_flg,
                expected_value=expected_value):
                self.assertEqual(AgentsNames.get_name(category, soft_update_flg, dueling_network_flg), expected_value)

if __name__ == "__main__":
    unittest.main()
