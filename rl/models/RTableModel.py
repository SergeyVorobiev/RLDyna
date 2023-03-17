from abc import abstractmethod
from typing import Any

from rl.models.RModel import RModel


class RTableModel(RModel):

    @abstractmethod
    def update_state_visit_count(self, state: Any, model_index: int = 0):
        ...

    @abstractmethod
    def update_action_visit_count(self, state: Any, action: int, model_index: int = 0):
        ...

    @abstractmethod
    def reset_actions_visit(self, model_index: int = 0):
        ...

    @abstractmethod
    def is_action_visited(self, state, action, model_index: int = 0):
        ...

    @abstractmethod
    def get_action_visit_count(self, state: Any, action: int, model_index: int = 0):
        ...

    @abstractmethod
    def get_state_visit_count(self, state: Any, model_index: int = 0):
        ...
