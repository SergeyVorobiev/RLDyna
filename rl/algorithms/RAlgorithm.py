from abc import abstractmethod
from typing import Any

from rl.models.RModel import RModel
from rl.planning.RLearnMemory import RLearnMemory
from rl.policy.RPolicy import RPolicy


# The base class of QAlgorithm, SARSAAlgorithm, MC and others.
# The algorithm itself usually uses the 1 memory capacity because it learns from what it saw just right now.
# Learning process from the memory is called planning.
class RAlgorithm(RLearnMemory):

    # task_complete_checker is intended to check if the task has really been accomplished, or it has been ended
    # by timeout, it is important because in case if task ends by timeout you probably want to reset discount but
    # as the task still has not been done you probably want to add U' or Q' to the reward, either if the task is done
    # you just add the reward because there is no potential U' or Q' further
    def __init__(self, policy: RPolicy = None, alpha: float = 1, discount: float = 1, memory_capacity: int = 1,
                 terminal_state_checker=None):
        super().__init__(memory_capacity)
        self._alpha: float = alpha
        self._discount: float = discount
        self._policy: RPolicy = policy
        if terminal_state_checker is None:
            self._terminal_state_checker = self._default_terminal_state_checker
        else:
            self._terminal_state_checker = terminal_state_checker
        self._done: bool = False

    @staticmethod
    def _default_terminal_state_checker(truncated, props):
        return not truncated

    # returns - error, batch
    @abstractmethod
    def train(self, models: [RModel]) -> (float, Any):
        ...

    # for planning. returns - error, batch
    @abstractmethod
    def plan(self, models: [RModel], batch) -> (float, Any):
        ...

    @abstractmethod
    def get_q_values(self, models: [RModel], state: Any):
        ...

    @abstractmethod
    def get_action_values(self, models: [RModel], state: Any):
        ...

    @abstractmethod
    def get_v(self, models: [RModel], state: Any) -> float:
        ...

    @abstractmethod
    def pick_action(self, models: [RModel], state: Any) -> Any:
        ...

    @abstractmethod
    def update_policy(self):
        ...

    def get_policy(self) -> RPolicy:
        return self._policy
