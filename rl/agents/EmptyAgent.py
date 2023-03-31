from gym import Env
from gym.spaces import Discrete, Box

from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.algorithms.EmptyAlgorithm import EmptyAlgorithm
from rl.dyna.Dyna import Dyna


class EmptyAgent(RDynaAgentBuilder):

    def __init__(self):
        pass

    def build_agent(self, env: Env):
        if type(env.action_space) == Discrete:
            action = 0
        elif type(env.action_space) == Box:
            space: Box = env.action_space
            action = space.low
        else:
            raise NotImplementedError("Action space is unknown: " + str(type(env.action_space)))
        return Dyna(models=[], algorithm=EmptyAlgorithm(action))
