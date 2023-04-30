from typing import Any

from rl.models.RModel import RModel
from rl.algorithms.RAlgorithm import RAlgorithm
from rl.rewardestimators.RewardEstimator import RewardEstimator
from rl.stateprepares.RStatePrepare import RStatePrepare
from rl.planning.RPlanning import RPlanning


class Dyna(object):

    def __init__(self, models: [RModel], algorithm: RAlgorithm, state_prepare: RStatePrepare = None,
                 reward_estimator: RewardEstimator = None,
                 planning: RPlanning = None, allow_clear_memory=True, test_mode=False):
        self._models: [RModel] = models
        self._algorithm: RAlgorithm = algorithm
        self._state_prepare = state_prepare
        self._planning: RPlanning = planning
        self._allow_clear_memory = allow_clear_memory
        self._reward_estimator: RewardEstimator = reward_estimator
        self._test_mode = test_mode

    def clear_memory(self):
        if self._planning is not None and self._allow_clear_memory:
            self._planning.clear_memory()

    def get_algorithm_memory_capacity(self):
        return self._algorithm.get_memory_capacity()

    def get_models(self):
        return self._models

    def act(self, state):
        state = self._prepare_raw_state(state)
        return self._algorithm.pick_action(self._models, state)

    def learn(self, state, action, reward, next_state, done, truncated, env_props):
        if self._test_mode:
            return
        p_state = self._prepare_raw_state(state)
        p_next_state = self._prepare_raw_state(next_state)
        if self._reward_estimator is not None:
            reward = self._reward_estimator.estimate(state, action, reward, next_state, done, truncated, env_props)
        self._algorithm.memorize_step(p_state, action, reward, p_next_state, done, truncated, env_props)
        error, batch = self._algorithm.train(self._models)
        if self._planning is not None and batch is not None:
            self._planning.memorize((error, batch))
            self._planning.plan(self._models, self._algorithm)
        self._algorithm.update_policy()

    def _prepare_raw_state(self, state) -> Any:
        if self._state_prepare is None:
            return state
        return self._state_prepare.prepare_raw_state(state)

    def get_v(self, state):
        state = self._prepare_raw_state(state)
        return self._algorithm.get_v(self._models, state)

    def improve_policy(self):
        policy = self._algorithm.get_policy()
        if policy is not None:
            policy.improve()

    def get_q_values(self, state):
        state = self._prepare_raw_state(state)
        return self._algorithm.get_q_values(self._models, state)

    def get_action_values(self, state):
        state = self._prepare_raw_state(state)
        return self._algorithm.get_action_values(self._models, state)
