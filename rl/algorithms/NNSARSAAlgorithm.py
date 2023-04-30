from typing import Any

import numpy as np

from rl.algorithms.RAlgorithm import RAlgorithm
from rl.algorithms.helpers.MCHelper import MCHelper
from rl.models.RModel import RModel
from rl.policy.RPolicy import RPolicy


# On Policy, Transforms to MC if N = episode's steps
class NNSARSAAlgorithm(RAlgorithm):

    def __init__(self, policy: RPolicy, alpha: float, discount: float, memory_capacity=1, terminal_state_checker=None,
                 next_q_max=False, clear_memory_every_n_steps=False, actions_listener=None):
        super().__init__(policy, alpha, discount, memory_capacity=memory_capacity,
                         terminal_state_checker=terminal_state_checker)
        self._actions_listener = actions_listener
        self._next_action = None
        self._next_q_max = next_q_max  # Just Q algorithm
        self._clear_memory_every_n_steps = clear_memory_every_n_steps
        self._x = []
        self._y = []

    def get_action_values(self, models: [RModel], state: Any):
        pass

    def plan(self, models: [RModel], batch) -> (float, Any):
        pass

    def update_policy(self):
        pass

    def pick_action(self, models: [RModel], state: Any) -> int:
        if self._actions_listener is not None:
            self._actions_listener(self.get_q_values(models, state))

        # When we last time calculate the next action for the algorithm we need to pick it up
        if self._next_action is not None:
            next_action = self._next_action
            self._next_action = None
            return next_action
        return self._policy.pick(self.get_q_values(models, state))

    def get_v(self, models: [RModel], state: Any):
        pass

    def get_q_values(self, models: [RModel], state: Any):
        return models[0].get_q_values(state)

    def train(self, models: [RModel]) -> (float, Any):
        _, _, _, next_state, done, truncated, props = self.get_last_memorized()[0]
        if self.is_memory_full() or done:
            used_tail = False
            model: RModel = models[0]
            batch = self.get_last_memorized(self.get_memory_size())  # Get all from end to start
            qs = model.get_q_values(next_state)
            if self._next_q_max:
                self._next_action = np.argmax(qs)  # Q
            else:
                self._next_action = self.get_policy().pick(qs)  # SARSA

            next_q = qs[self._next_action]

            if done:
                self._next_action = None

            # Check that the terminal was achieved if done
            if done and self._terminal_state_checker(truncated, props):
                start_g = 0  # If done we do not have a tail
            else:
                start_g = next_q  # if we are not done yet then attach the tail Q'
                used_tail = True
            result = MCHelper.build_g(batch, self._discount, start_g, need_reverse=True, used_tail=used_tail)
            batch.reverse()

            k = 0

            # Start from start to end
            for state, action, reward, next_state, done, truncated, props in batch:
                self.pick_data(state, action, reward, next_state, done, truncated, float(result[k][0]),
                               float(result[k][1]), props)
                k += 1
            self.send_data_to_model(models)
            self._x.clear()
            self._y.clear()
            if self._clear_memory_every_n_steps:
                self.clear_memory()
        return 0, None

    def pick_data(self, state, action, reward, next_state, done, truncated, state_g, state_discount, props):
        self._x.append(state)

        self._y.append([state_g, float(action), float(done)])

    def send_data_to_model(self, models):
        model: RModel = models[0]
        model.update(data=(self._x, self._y), batch_size=self._x.__len__())
