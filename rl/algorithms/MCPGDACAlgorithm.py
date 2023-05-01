from typing import Any

from rl.algorithms.MCACAlgorithm import MCACAlgorithm
from rl.models.RModel import RModel


# Monte Carlo Policy-Gradient discrete + Critic baseline
class MCPGDACAlgorithm(MCACAlgorithm):

    def __init__(self, discount: float = 1, memory_capacity: int = 1, tail_method=None,
                 clear_memory_every_n_steps=None, terminal_state_checker=None, actions_listener=None):
        super().__init__(discount=discount, memory_capacity=memory_capacity, tail_method=tail_method,
                         clear_memory_every_n_steps=clear_memory_every_n_steps,
                         terminal_state_checker=terminal_state_checker)
        self._actions_listener = actions_listener
        self._baseline_before = False
        self._x = []
        self._y = []
        self._y2 = []

    def pick_data(self, state, action, reward, next_state, done, truncated, state_g, state_discount, props):
        self._x.append(state)
        self._y.append([state_g, action, state_discount, done])
        self._y2.append([state_g, done])

    def pick_action(self, models: [RModel], state: Any) -> Any:
        if self._actions_listener is not None:
            self._actions_listener(self.get_action_values(models, state))
        return models[0].get_max_a(state)
