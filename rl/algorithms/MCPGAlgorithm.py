from typing import Any

from rl.algorithms.MCAlgorithm import MCAlgorithm
from rl.models.RModel import RModel


# Monte Carlo Policy-Gradient control
# This algorithm shows how we can train agent avoiding usage of p and q values
class MCPGAlgorithm(MCAlgorithm):

    def __init__(self, discount: float = 1, memory_capacity: int = 1):
        super().__init__(alpha=1, discount=discount, memory_capacity=memory_capacity)
        self._x = []
        self._y = []

    def pick_data(self, state, action, reward, next_state, done, state_g, state_discount, props):
        self._x.append(state)
        self._y.append([action, state_g])

    def send_data_to_model(self, models):

        # We have values from end to start, technically the order does not matter, as we anyway has the complete
        # episode
        models[0].update((self._x, self._y))
        self._x.clear()
        self._y.clear()

    def get_a_distribution(self, models: [RModel], state: Any):
        return models[0].get_a_distribution(state)
