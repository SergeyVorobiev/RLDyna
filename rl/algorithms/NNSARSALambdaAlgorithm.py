from typing import Any
from rl.algorithms.RAlgorithm import RAlgorithm
from rl.models.RModel import RModel
from rl.policy.RPolicy import RPolicy


# On Policy
class NNSARSALambdaAlgorithm(RAlgorithm):

    def __init__(self, policy: RPolicy, alpha: float, discount: float, n_step: int = 1):
        super().__init__(policy, alpha, discount, n_step)
        self._n_step = n_step
        self._next_action = None

    def get_a_distribution(self, models: [RModel], state: Any):
        pass

    def plan(self, models: [RModel], batch) -> (float, Any):
        pass

    def update_policy(self):
        pass

    def pick_action(self, models: [RModel], state: Any) -> int:
        if self._next_action is not None:
            return self._next_action
        return self._policy.pick(self.get_q_values(models, state))

    def get_v(self, models: [RModel], state: Any):
        pass

    def get_q_values(self, models: [RModel], state: Any):
        return models[0].get_q_values(state)

    def train(self, models: [RModel]) -> (float, Any):
        _, _, _, _, done, _ = self.get_last_memorized()[0]
        if self.is_memory_full() or done:
            x = []
            y = []
            model: RModel = models[0]
            batch = self.get_last_memorized(self.get_memory_size(), last_first=False)  # Get all from start to end
            g = 0
            length = batch.__len__()
            k = 0
            for state, action, reward, next_state, done, _ in batch:
                next_qs = model.get_q_values(next_state)
                g = g + reward
                if (k + 1) < length:
                    _, action, reward, _, _, _ = batch[k + 1]
                    next_q = g
                else:
                    if not done:
                        self._next_action = self.get_policy().pick(next_qs)
                        next_q = next_qs[self._next_action]
                    else:
                        self._next_action = None
                        next_q = g
                x.append(state)
                y.append([float(next_q), float(g), float(action), float(done), float(self._discount)])
            model.update(data=(x, y))
            self.clear_memory()
        return 0, None
