from typing import Any
from rl.algorithms.StepControl import StepControl
from rl.models.RModel import RModel


# on policy TD control
class ExpectedSARSA(StepControl):

    def update_policy(self):
        pass

    def train_sample(self, models: [RModel], state: Any, action: int, reward: float, next_state: Any,
                     done: bool, env_props: Any) -> float:
        q = models[0].get_q(state, action)
        g = reward
        if not done:
            next_q_values = models[0].get_q_values(next_state)
            result = 0
            for a in range(next_q_values.__len__()):
                next_q = models[0].get_q(next_state, a) * self._policy.get_action_probability(next_q_values, a)
                result += next_q
            g = g + self._discount * result
        error = g - q
        q = q + self._alpha * error
        models[0].update_q(state, action, q, done)
        return abs(error)

    def get_q_values(self, models: [RModel], state: Any):
        return models[0].get_q_values(state)

    def get_v(self, models: [RModel], state: Any) -> float:
        pass

    def pick_action(self, models: [RModel], state: Any) -> int:
        return self._policy.pick(self.get_q_values(models, state))
