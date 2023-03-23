import random

import numpy as np

from rl.policy.RPolicy import RPolicy


class EGreedyRPolicy(RPolicy):

    def __init__(self, epsilon: float, threshold: float = 0, improve_step: float = 0):
        self._epsilon: float = epsilon
        self._threshold: float = threshold
        self._improve_step: float = improve_step

    def pick(self, values):
        if random.uniform(0, 1) < self._epsilon:
            return random.randrange(0, values.__len__())
        max_value = np.amax(values)
        max_actions = np.argwhere(values == max_value).flatten()
        return random.choice(max_actions)

    def get_action_probability(self, values, action) -> float:
        any_action_probability = 1 / values.__len__() * self._epsilon
        max_value = np.amax(values)
        max_actions = np.argwhere(values == max_value).flatten()
        max_action_epsilon = 1 - self._epsilon
        if max_action_epsilon == 0:
            max_action_probability = any_action_probability
        else:
            max_action_probability = max_action_epsilon / max_actions.__len__() + any_action_probability
        if max_actions.__contains__(action):
            return max_action_probability
        return any_action_probability

    def improve(self):
        if self._epsilon > self._threshold:
            self._epsilon -= self._improve_step




