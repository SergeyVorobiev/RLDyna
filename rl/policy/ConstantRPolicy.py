import random
from abc import ABC

from rl.policy.RPolicy import RPolicy


class ConstantRPolicy(RPolicy, ABC):

    def __init__(self, action: int, epsilon: float):
        self.__epsilon: float = epsilon
        self.__action: int = action

    def pick(self, values):
        if random.uniform(0, 1) < self.__epsilon:
            return random.randrange(0, values.__len__())
        return self.__action

    def get_action_probability(self, values, action) -> float:
        prob = 1 / values.__len__()
        if self.__action == action:
            return 1 - self.__epsilon + self.__epsilon * prob
        else:
            return self.__epsilon * prob



