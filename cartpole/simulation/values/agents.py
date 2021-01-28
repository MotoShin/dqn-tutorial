from enum import Enum


class AgentsNames(Enum):
    DQN = "dqn"
    DQNSOFTUPDATE = "dqn_softupdate"
    DDQN = "ddqn"
    DDQNDUELINGNET = "ddqn_dueling_network"

    @classmethod
    def value_of(cls, target_value):
        for name in AgentsNames:
            if name.value == target_value:
                return name
        raise ValueError('{} is not found'.format(target_value))