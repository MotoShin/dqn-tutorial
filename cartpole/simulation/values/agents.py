from enum import Enum


class AgentsNames(Enum):
    DQN = "dqn"
    DQN_SOFTUPDATE = "dqn_softupdate"
    DQN_DUELINGNET = "dqn_dueling_network"
    DQN_SOFTUPDATE_DUELINGNET = "dqn_softupdate_dueling_network"
    DDQN = "ddqn"
    DDQN_SOFTUPDATE = "ddqn_softupdate"
    DDQN_DUELINGNET = "ddqn_dueling_network"
    DDQN_SOFTUPDATE_DUELINGNET = "ddqn_softupdate_dueling_network"

    @classmethod
    def value_of(cls, target_value):
        for name in AgentsNames:
            if name.value == target_value:
                return name
        raise ValueError('{} is not found'.format(target_value))
