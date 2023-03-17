from typing import Any

from rl.models.RModel import RModel
from rl.algorithms.RAlgorithm import RAlgorithm
from rl.planning.RPlanning import RPlanning


class Dyna(object):

    def __init__(self, models: [RModel], algorithm: RAlgorithm, state_preparator, planning: RPlanning = None):
        self._models: [RModel] = models
        self._algorithm: RAlgorithm = algorithm
        self._state_preparator = state_preparator
        self._planning: RPlanning = planning

    def act(self, state):
        state = self.prepare_raw_state(state)
        return self._algorithm.pick_action(self._models, state)

    def learn(self, state, action, reward, next_state, done, env_props):
        state = self.prepare_raw_state(state)
        next_state = self.prepare_raw_state(next_state)
        self._algorithm.memorize_step(state, action, reward, next_state, done, env_props)
        error, batch = self._algorithm.train(self._models)
        if self._planning is not None and batch is not None:
            self._planning.memorize((error, batch))
            self._planning.plan(self._models, self._algorithm)
        self._algorithm.update_policy()

    def prepare_raw_state(self, state) -> Any:
        return self._state_preparator.prepare_raw_state(state)

    def get_v(self, state):
        state = self.prepare_raw_state(state)
        return self._algorithm.get_v(self._models, state)

    def get_q_values(self, state):
        state = self.prepare_raw_state(state)
        return self._algorithm.get_q_values(self._models, state)
