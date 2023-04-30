from typing import Any

from rl.algorithms.MCAlgorithm import MCAlgorithm
from rl.models.RModel import RModel


# Monte Carlo Policy-Gradient control
# This algorithm shows how we can train agent avoiding usage of p and q values
class MCPGDAlgorithm(MCAlgorithm):

    def __init__(self, discount: float = 1, memory_capacity: int = 1, tail_method=None):
        super().__init__(alpha=1, discount=discount, memory_capacity=memory_capacity, tail_method=tail_method)
        self._x = []
        self._y = []

    def pick_data(self, state, action, reward, next_state, done, truncated, state_g, state_discount, props):
        self._x.append(state)
        self._y.append([action, state_g, state_discount, done])

    def send_data_to_model(self, models):
        models[0].update((self._x, self._y), self._x.__len__())
        self._x.clear()
        self._y.clear()

    def get_action_values(self, models: [RModel], state: Any):
        return models[0].get_action_values(state)
