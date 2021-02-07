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

    @staticmethod
    def get_name(category, soft_update_flg, dueling_network_flg):
        agent_name = None

        if category == AgentsNames.DQN.value:
            if soft_update_flg and dueling_network_flg:
                agent_name = AgentsNames.DQN_SOFTUPDATE_DUELINGNET.value
            elif soft_update_flg:
                agent_name = AgentsNames.DQN_SOFTUPDATE.value
            elif dueling_network_flg:
                agent_name = AgentsNames.DQN_DUELINGNET.value
            else:
                agent_name = AgentsNames.DQN.value

        if category == AgentsNames.DDQN.value:
            if soft_update_flg and dueling_network_flg:
                agent_name = AgentsNames.DDQN_SOFTUPDATE_DUELINGNET.value
            elif soft_update_flg:
                agent_name = AgentsNames.DDQN_SOFTUPDATE.value
            elif dueling_network_flg:
                agent_name = AgentsNames.DDQN_DUELINGNET.value
            else:
                agent_name = AgentsNames.DDQN.value

        return agent_name

    @classmethod
    def value_of(cls, target_value):
        for name in AgentsNames:
            if name.value == target_value:
                return name
        raise ValueError('{} is not found'.format(target_value))
