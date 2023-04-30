from abc import ABC
from typing import Any

from rl.models.RModel import RModel


# Model that compute policy not basing on q and v directly.
class PolicyGradientAbsModel(RModel, ABC):

    def __init__(self, n_actions):
        super().__init__(n_actions)

    def get_q_values(self, state: Any, model_index: int = 0):
        raise NotImplementedError

    def get_max_q(self, state: Any, model_index: int = 0):
        raise NotImplementedError

    def get_q(self, state, action: int, model_index: int = 0):
        raise NotImplementedError

    def get_v(self, state: Any, model_index: int = 0):
        raise NotImplementedError

    def update_q(self, state: Any, action: int, q: float, episode_done: bool, model_index: int = 0):
        raise NotImplementedError

    def update_v(self, state: Any, v: float, episode_done: bool, model_index: int = 0):
        raise NotImplementedError

    def update_q_values(self, state: Any, values, episode_done: bool, model_index: int = 0):
        raise NotImplementedError
