import numpy as np

from typing import Any

from rl.collections.StateHashBank import StateHashBank
from rl.models.RModel import RModel
from rl.models.nnbuilders.NNGridBuilder import NNGridBuilder


class ObstacleModel(RModel):

    def get_a_distribution(self, state: Any, model_index: int = 0):
        raise NotImplementedError

    def __init__(self, input_shape, n_actions, batch_size, epochs, steps_to_train=200):
        super().__init__(n_actions)
        self._input_shape = input_shape
        self._n_actions = n_actions
        self._steps_to_train = steps_to_train
        self._model = NNGridBuilder.build_frozen_lake_around_supporter()
        self._state_hash_bank = StateHashBank(20)
        self._steps = 0
        self._batch_size = batch_size
        self._epochs = epochs

    def get_max_q(self, state: Any, model_index: int = 0):
        raise Exception("Not implemented")

    def get_q(self, state, action: int, model_index: int = 0):
        raise Exception("Not implemented")

    def get_v(self, state: Any, model_index: int = 0):
        raise Exception("Not implemented")

    def update_q(self, state: Any, action: int, q: float, episode_done: bool, model_index: int = 0):
        self._steps = self._steps + 1
        self._add_to_batch(state, action, q)
        if self._steps >= self._steps_to_train or episode_done:
            self._train_batch()

    def _add_to_batch(self, state: Any, action: int, q: float):
        hash_value = self.get_state_hash(state)
        qs1 = self.get_q_values(state)
        _ = self._state_hash_bank.update(hash_value, state[1], q, action, qs1)[0]

    def _train_batch(self):
        self._steps = 0
        qs, states, _ = self._state_hash_bank.get_values()
        bs = np.array(states)
        bq = np.array(qs)
        print("Obstacle: " + str(bs.__len__()))
        self._model.fit(bs, bq, epochs=self._epochs, verbose=0)

    def update_v(self, state: Any, v: float, episode_done: bool, model_index: int = 0):
        raise Exception("Not implemented")

    def update_q_values(self, state: Any, values, episode_done: bool, model_index: int = 0):
        raise Exception("Not implemented")

    def get_max_a(self, state: Any, model_index: int = 0):
        raise Exception("Not implemented")

    def get_state_hash(self, state) -> Any:
        return hash(state[1].data.tobytes())

    def get_q_values(self, state: Any, model_index: int = 0):
        q_values = self._model(np.array([state[1]]), training=False)
        return q_values[0].numpy()

    def update(self, data: Any):
        raise NotImplementedError

    def save(self, path=None) -> (bool, str):
        return False, None
