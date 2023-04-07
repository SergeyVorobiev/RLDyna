from typing import Any

from rl.algorithms.MCAlgorithm import MCAlgorithm
from rl.models.RModel import RModel


# Monte Carlo Policy-Gradient + TD baseline
class MCPGTDAlgorithm(MCAlgorithm):

    def __init__(self, discount: float = 1, memory_capacity: int = 1):
        super().__init__(alpha=1, discount=discount, memory_capacity=memory_capacity)
        self._x = []
        self._y = []
        self._y2 = []

    def pick_data(self, state, action, reward, next_state, done, state_g, state_discount, props):
        self._x.append(state)
        self._y.append([action, state_g, state_discount])
        self._y2.append([state_g])

    def send_data_to_model(self, models):

        # Train Critic
        models[1].update((self._x, self._y2))

        # Get predicted G
        predictions = models[1].predict(self._x)

        # Replace G by G - predictions (this is our baseline)
        size = self._y.__len__()
        for i in range(size):
            self._y[i][1] = self._y[i][1] - float(predictions[0])

        # Update policy
        models[0].update((self._x, self._y))
        self._x.clear()
        self._y.clear()
        self._y2.clear()

    def get_a_distribution(self, models: [RModel], state: Any):
        return models[0].get_a_distribution(state)
