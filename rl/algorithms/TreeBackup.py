from typing import Any

from rl.algorithms.RAlgorithm import RAlgorithm
from rl.models.RModel import RModel
from rl.policy.RPolicy import RPolicy


class TreeBackup(RAlgorithm):

    def __init__(self, policy: RPolicy, alpha: float, discount: float, n_step: int = 1):
        super().__init__(policy, alpha, discount, n_step)
        self._n_step = n_step

    def plan(self, models: [RModel], batch) -> (float, Any):
        return self.train_from_past(models[0], batch)

    def train(self, models: [RModel]) -> (float, Any):
        return self.train_from_past(models[0], self.get_last_memorized(size=self._n_step))

    def pick_action(self, models: [RModel], state: Any) -> int:
        return self._policy.pick(self.get_q_values(models, state))

    def get_v(self, models: [RModel], state: Any) -> float:
        pass

    def get_q_values(self, models: [RModel], state: Any):
        return models[0].get_q_values(state)

    def update_policy(self):
        pass

    def __get_expected_g(self, next_q_values: [], except_a: int = None):
        result = 0
        for a in range(next_q_values.__len__()):
            if except_a is not None and except_a == a:
                continue
            next_q = next_q_values[a] * self._policy.get_action_probability(next_q_values, a)
            result += next_q
        return result

    def train_from_past(self, model: RModel, batch: Any) -> (float, Any):
        n = batch.__len__()
        state, action, reward, next_state, done, _ = batch[0]
        g = reward
        expected_g = 0
        action_probability = 0
        if not done:
            expected_g = self.__get_expected_g(model.get_q_values(next_state))
            g = g + self._discount * expected_g
        q = model.get_q(state, action)
        error = g - q
        q = q + self._alpha * error
        error = abs(error)
        if n > 1:
            q_values = model.get_q_values(state)
            action_probability = self._policy.get_action_probability(q_values, action)
            expected_g = self.__get_expected_g(q_values, action)
        model.update_q(state, action, q, done)
        for i in range(1, n):
            state, action, reward, next_state, done, _ = batch[i]
            cur_g = reward + self._discount * expected_g
            g = cur_g + action_probability * g
            q_values = model.get_q_values(state)
            action_probability = self._policy.get_action_probability(q_values, action)
            expected_g = self.__get_expected_g(q_values, action)
            q = model.get_q(state, action)
            cur_error = g - q
            error = error + abs(cur_error)
            q = q + self._alpha * cur_error
            model.update_q(state, action, q, done)
        return error, batch
