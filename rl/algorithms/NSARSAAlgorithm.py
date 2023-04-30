from typing import Any
from rl.algorithms.RAlgorithm import RAlgorithm
from rl.models.RModel import RModel
from rl.policy.RPolicy import RPolicy


# on policy TD control n-step
class NSARSAAlgorithm(RAlgorithm):

    def __init__(self, policy: RPolicy, alpha: float, discount: float, n_step: int = 1):
        super().__init__(policy, alpha, discount, n_step)
        self._n_step = n_step

    def get_action_values(self, models: [RModel], state: Any):
        pass

    def plan(self, models: [RModel], batch) -> (float, Any):
        return self.train_from_past(models[0], batch)

    def update_policy(self):
        pass

    def pick_action(self, models: [RModel], state: Any) -> int:
        return self._policy.pick(self.get_q_values(models, state))

    def get_v(self, models: [RModel], state: Any):
        pass

    def get_q_values(self, models: [RModel], state: Any):
        return models[0].get_q_values(state)

    def train(self, models: [RModel]) -> (float, Any):
        return self.train_from_past(models[0], self.get_last_memorized(size=self._n_step))

    def train_from_past(self, model: RModel, batch: Any) -> (float, Any):
        model = model
        n = batch.__len__()
        state, action, reward, next_state, done, truncated, _ = batch[0]
        g = pow(self._discount, n - 1) * reward
        if not done:
            a = self._policy.pick(model.get_q_values(next_state))
            next_q = model.get_q(next_state, a)
            g = g + pow(self._discount, n) * next_q
        q = model.get_q(state, action)
        error = g - q
        q = q + self._alpha * error
        error = abs(error)
        model.update_q(state, action, q, done)
        for i in range(1, n):
            state, action, reward, _, _, _, _ = batch[i]
            discounted_reward = pow(self._discount, n - 1 - i) * reward
            g = g + discounted_reward
            q = model.get_q(state, action)
            cur_error = g - q
            error = error + abs(cur_error)
            q = q + self._alpha * cur_error
            model.update_q(state, action, q, done)
        return error, batch
