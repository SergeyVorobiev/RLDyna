from abc import abstractmethod
from typing import Any


class RModel(object):

    def __init__(self, n_actions):
        self._n_actions = n_actions

    @abstractmethod
    def get_q_values(self, state: Any, model_index: int = 0):
        ...

    @abstractmethod
    def get_max_q(self, state: Any, model_index: int = 0):
        ...

    @abstractmethod
    def get_q(self, state, action: int, model_index: int = 0):
        ...

    @abstractmethod
    def get_v(self, state: Any, model_index: int = 0):
        ...

    @abstractmethod
    def update_q(self, state: Any, action: int, q: float, episode_done: bool, model_index: int = 0):
        ...

    @abstractmethod
    def update_v(self, state: Any, v: float, episode_done: bool, model_index: int = 0):
        ...

    @abstractmethod
    def update_q_values(self, state: Any, values, episode_done: bool, model_index: int = 0):
        ...

    @abstractmethod
    def get_max_a(self, state: Any, model_index: int = 0):
        ...

    @abstractmethod
    def get_state_hash(self, state) -> Any:
        ...
