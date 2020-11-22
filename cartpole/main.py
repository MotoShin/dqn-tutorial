import sys

from simulation.simulate import Simulate


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        agent = 'dqn'
    else:
        agent = args[1]
    Simulate(agent).start()
