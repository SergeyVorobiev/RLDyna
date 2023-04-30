from typing import Any

import tensorflow
from tensorflow_probability.python.distributions import Normal

from rl.algorithms.MCACAlgorithm import MCACAlgorithm
from rl.models.RModel import RModel


# Monte Carlo Policy-Gradient Continuous + Critic
class MCPGCACAlgorithm(MCACAlgorithm):

    def __init__(self, discount: float = 1, memory_capacity: int = 1, gauss_listener=None, batch_size=0,
                 clear_memory_every_n_steps=False, tail_method=None, terminal_state_checker=None):
        super().__init__(discount=discount, memory_capacity=memory_capacity, batch_size=batch_size,
                         clear_memory_every_n_steps=clear_memory_every_n_steps, tail_method=tail_method,
                         terminal_state_checker=terminal_state_checker)
        self._gauss_listener = gauss_listener

    def pick_data(self, state, action, reward, next_state, done, truncated, state_g, state_discount, props):
        self._x.append(state)
        data = [state_g, state_discount, done]
        for a in action:
            data.append(a)
        self._y.append(data)
        self._y2.append([state_g, done])

    def pick_action(self, models: [RModel], state: Any) -> Any:
        result = models[0].predict([state])
        means = result[0][0]
        deviations = result[1][0]
        actions = Normal(loc=means, scale=deviations).sample()
        if self._gauss_listener is not None:
            self._gauss_listener(means, deviations, actions)
        return actions
