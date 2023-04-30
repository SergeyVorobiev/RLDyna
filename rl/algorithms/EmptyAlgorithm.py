from typing import Any

from rl.algorithms.RAlgorithm import RAlgorithm
from rl.models.RModel import RModel
from rl.policy.ConstantRPolicy import ConstantRPolicy


class EmptyAlgorithm(RAlgorithm):

    def __init__(self, action):
        super().__init__(ConstantRPolicy(0, 0), 1, 1, 1)
        self._action = action

    def train(self, models: [RModel]) -> (float, Any):
        return 0, 0

    def plan(self, models: [RModel], batch) -> (float, Any):
        return 0, 0

    def get_q_values(self, models: [RModel], state: Any):
        return 0

    def get_v(self, models: [RModel], state: Any) -> float:
        return 0

    def pick_action(self, models: [RModel], state: Any) -> Any:
        return self._action

    def update_policy(self):
        pass

    def get_action_values(self, models: [RModel], state: Any):
        pass
