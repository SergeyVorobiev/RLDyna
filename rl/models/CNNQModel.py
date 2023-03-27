from collections import deque

import keras
import numpy as np

from typing import Any

from keras.initializers.initializers_v2 import VarianceScaling, Constant

from rl.collections.StateHashBank import StateHashBank
from rl.models.RModel import RModel
from rl.models.nnbuilders.NNGridBuilder import NNGridBuilder


class CNNQModel(RModel):

    def save(self, path=None):
        if path is not None:
            self._support_model.save(path)
        elif self._model_path is not None:
            self._support_model.save(self._model_path)

    def __init__(self, input_shape, n_actions, batch_size, epochs, hash_unique_states_capacity,
                 steps_to_train=200, model_path=None, load_model=False):
        super().__init__(n_actions)
        self._input_shape = input_shape
        self._n_actions = n_actions
        self._steps_to_train = steps_to_train
        self._input_shape = input_shape
        self._model = None
        self._support_model = None
        self._model_path = model_path
        self._load_model = load_model
        self._load()

        self._unique_states_size = hash_unique_states_capacity

        # Updates for unique states
        self._state_hash_bank = StateHashBank(self._unique_states_size)

        # General storage that will keep both unique and regular values to diverse as much as possible
        # Not used currently
        self._batch_states: deque = deque(maxlen=batch_size)
        self._batch_qs: deque = deque(maxlen=batch_size)
        self._steps = 0
        self._batch_size = batch_size
        self._epochs = epochs
        self._last_state = None
        self._epochs_for_little_train = 5
        self._koef1 = 0.8
        self._koef2 = 0.2

    def get_max_q(self, state: Any, model_index: int = 0):
        hash_value = self.get_state_hash(state)
        value = self._state_hash_bank.get(hash_value)
        if value is None:
            return self._get_q_values_from_support(state).max() * self._koef1 + self._get_q_values_from_main(
                state).max() * self._koef2
        return max(value[0]) * self._koef1 + self._get_q_values_from_main(state).max() * self._koef2

    def get_q(self, state, action: int, model_index: int = 0):
        hash_value = self.get_state_hash(state)
        value = self._state_hash_bank.get(hash_value)
        if value is None:
            return self._get_q_values_from_support(state)[action] * self._koef1 + self._get_q_values_from_main(state)[
                action] * self._koef2
        return value[0][action] * self._koef1 + self._get_q_values_from_main(state)[action] * self._koef2

    def get_v(self, state: Any, model_index: int = 0):
        raise Exception("Not implemented")

    def _train_unique_states(self, epochs):
        qs, states, _ = self._state_hash_bank.get_values()
        bq = np.array(qs)
        bs = np.array(states)
        self._model.fit(bs, bq, epochs=epochs, verbose=0)

    def update_q(self, state: Any, action: int, q: float, episode_done: bool, model_index: int = 0):
        self._steps = self._steps + 1

        hash_value = self.get_state_hash(state)
        self._add_to_batch(state, action, q, hash_value)

        if hash_value == self._last_state:
            self._train_unique_states(self._epochs_for_little_train)

        if self._steps >= self._steps_to_train or episode_done:
            self._train_batch()

        # Update support model
        if episode_done:
            self._support_model.set_weights(self._model.get_weights())
        self._last_state = hash_value

    def _add_to_batch(self, state: Any, action: int, q: float, hash_value: int):

        # We use only states with unique hash and suppose that every state has a unique hash
        # Technically we train NN by using tabular approach
        value = self._state_hash_bank.get(hash_value)
        qs_support = self._get_q_values_from_support(state)
        if value is None:
            qs2 = (qs_support + self._get_q_values_from_main(state)) / 2
            self._state_hash_bank.update(hash_value, state[0], q, action, qs2)
        else:
            qs2 = value[0]
            qs2[action] = q
            self._state_hash_bank.set(hash_value, state[0], qs2, action)
        return value

    def _train_batch(self):
        self._steps = 0
        qs, states, _ = self._state_hash_bank.get_values()
        bq = np.array(qs)
        bs = np.array(states)
        self._model.fit(bs, bq, epochs=self._epochs, verbose=0)

    def update_v(self, state: Any, v: float, episode_done: bool, model_index: int = 0):
        raise Exception("Not implemented")

    def update_q_values(self, state: Any, values, episode_done: bool, model_index: int = 0):
        raise Exception("Not implemented")

    def get_max_a(self, state: Any, model_index: int = 0):
        raise Exception("Not implemented")

    def get_state_hash(self, state) -> Any:
        return hash(state[0].data.tobytes())

    def get_q_values(self, state: Any, model_index: int = 0):
        hash_value = self.get_state_hash(state)
        value = self._state_hash_bank.get(hash_value)
        if value is None:
            return self._get_q_values_from_support(state) * self._koef1 + self._get_q_values_from_main(
                state) * self._koef2
        return value[0] * self._koef1 + self._get_q_values_from_main(state) * self._koef2

    def _get_q_values_from_main(self, state: Any):
        q_values = self._model(np.array([state[0]]), training=False)
        return q_values[0].numpy()

    def _get_q_values_from_support(self, state: Any):
        q_values = self._support_model(np.array([state[0]]), training=False)
        return q_values[0].numpy()

    def _load(self):
        k_init_main = VarianceScaling(distribution="uniform")
        k_init_support = Constant(0)
        self._model = NNGridBuilder.build_simple_frozen_lake_cnn(input_shape=self._input_shape,
                                                                 n_actions=self._n_actions,
                                                                 kernel_initializer=k_init_main)
        if self._model_path is not None and self._load_model:
            try:
                self._support_model = keras.models.load_model(self._model_path)
                self._model.set_weights(self._support_model.get_weights())
            except IOError as e:
                print("Model is not found: " + self._model_path)
                self._support_model = NNGridBuilder.build_simple_frozen_lake_cnn(input_shape=self._input_shape,
                                                                                 n_actions=self._n_actions,
                                                                                 kernel_initializer=k_init_support)
        else:
            self._support_model = NNGridBuilder.build_simple_frozen_lake_cnn(input_shape=self._input_shape,
                                                                             n_actions=self._n_actions,
                                                                             kernel_initializer=k_init_support)
