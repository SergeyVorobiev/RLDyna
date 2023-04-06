from typing import Any

from rl.algorithms.StepControl import StepControl
from rl.models.RModel import RModel


# off policy TD control
class QAlgorithm(StepControl):

    def get_v(self, models: [RModel], state: Any) -> float:
        pass

    def get_q_values(self, models: [RModel], state: Any):
        return models[0].get_q_values(state)

    def update_policy(self):
        pass

    # QAlgorithm(S,a) = QAlgorithm(S,a) + alpha * [ R + y * maxQ(S`, a) - QAlgorithm(S, a)]
    def train_sample(self, models: [RModel], state: Any, action: int, reward: float, next_state: Any,
                     done: bool, env_props: Any) -> float:
        q = models[0].get_q(state, action)
        g = reward
        if not done:
            next_max_q = models[0].get_max_q(next_state)
            g = reward + self._discount * next_max_q
        error = g - q
        q = q + self._alpha * error
        models[0].update_q(state, action, q, done)
        return abs(error)
