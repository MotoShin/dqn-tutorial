import sys

from simulation.simulate import Simulate
from simulation.values.agents import AgentsNames


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        agent = 'dqn'
    elif args[1] == 'help':
        print("agent name list")
        for agent_name in AgentsNames:
            print(agent_name.value)
        sys.exit()
    else:
        agent = args[1]
    Simulate(agent).start()
