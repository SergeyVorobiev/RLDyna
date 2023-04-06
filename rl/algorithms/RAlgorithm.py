from abc import abstractmethod
from typing import Any

from rl.models.RModel import RModel
from rl.planning.RLearnMemory import RLearnMemory
from rl.policy.RPolicy import RPolicy


# The base class of QAlgorithm, SARSAAlgorithm, MC and others.
# The algorithm itself usually uses the 1 memory capacity because it learns from what it saw just right now.
# Learning process from the memory is called planning.
class RAlgorithm(RLearnMemory):

    def __init__(self, policy: RPolicy = None, alpha: float = 1, discount: float = 1, memory_capacity: int = 1):
        super().__init__(memory_capacity)
        self._alpha: float = alpha
        self._discount: float = discount
        self._policy: RPolicy = policy
        self._done: bool = False

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
    def get_a_distribution(self, models: [RModel], state: Any):
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


