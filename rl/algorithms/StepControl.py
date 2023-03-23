from abc import abstractmethod
from typing import Any

from rl.models.RModel import RModel
from rl.policy.RPolicy import RPolicy
from rl.algorithms.RAlgorithm import RAlgorithm


class StepControl(RAlgorithm):

    def __init__(self, policy: RPolicy, alpha: float = 1, discount: float = 1):
        super().__init__(policy, alpha, discount, 1)

    def train(self, models: [RModel]) -> (float, Any):
        batch = self.get_last_memorized()
        state, action, reward, next_state, done, env_props = batch[0]
        return self.train_sample(models, state, action, reward, next_state, done, env_props), batch

    def plan(self, models: [RModel], batch) -> (float, Any):
        state, action, reward, next_state, done, env_props = batch[0]
        return self.train_sample(models, state, action, reward, next_state, done, env_props), batch

    @abstractmethod
    def train_sample(self, models: [RModel], state: Any, action: int, reward: float, next_state: Any,
                     done: bool, env_props: Any) -> float:
        ...

    def pick_action(self, models: [RModel], state: Any) -> int:
        return self._policy.pick(self.get_q_values(models, state))


