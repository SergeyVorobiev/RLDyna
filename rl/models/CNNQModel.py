import random
from collections import deque
from enum import Enum

import numpy as np

from typing import Any

from keras.initializers.initializers_v2 import VarianceScaling, Constant

from rl.models.RModel import RModel
from rl.models.nnbuilders.NNGridBuilder import NNGridBuilder


class MaxQ(Enum):
    from_support = 0
    from_main = 1
    mixed = 2


class CNNQModel(RModel):

    def __init__(self, input_shape, n_actions, batch_size, epochs, steps_to_train=200, max_q: MaxQ = MaxQ.from_support):
        super().__init__(n_actions)
        self._input_shape = input_shape
        self._n_actions = n_actions
        self._steps_to_train = steps_to_train
        k_init_main = VarianceScaling(distribution="uniform")
        k_init_support = Constant(0)
        self._model = NNGridBuilder.build_simple_frozen_lake_cnn(input_shape=input_shape, n_actions=n_actions,
                                                                 kernel_initializer=k_init_main)
        self._support_model = NNGridBuilder.build_simple_frozen_lake_cnn(input_shape=input_shape, n_actions=n_actions,
                                                                         kernel_initializer=k_init_support)
        self._batch_states: deque = deque(maxlen=batch_size)
        self._batch_qs: deque = deque(maxlen=batch_size)
        self._steps = 0
        self._batch_size = batch_size
        self._max_q_from = max_q
        self._epochs = epochs

    def get_max_q(self, state: Any, model_index: int = 0):
        if self._max_q_from == MaxQ.from_support:
            return self._get_q_values_from_support(state).max()
        elif self._max_q_from == MaxQ.from_main:
            return self._get_q_values_from_main(state).max()
        else:
            return (self._get_q_values_from_main(state).max() + self._get_q_values_from_support(state).max()) / 2

    def get_q(self, state, action: int, model_index: int = 0):
        return self._get_q_values_from_main(state)[action]

    def get_v(self, state: Any, model_index: int = 0):
        raise Exception("Not implemented")

    def update_q(self, state: Any, action: int, q: float, episode_done: bool, model_index: int = 0):
        self._steps = self._steps + 1

        # We do not train model after each q-updating, we form batch of samples, and after it is filled up we train.
        # Be careful to train nn after each step, it just unsets the weights and possibly diverge.
        # NN should see the whole or very variate picture to be able to tune weights correctly
        self._batch_states.append(state)

        qs1 = self._get_q_values_from_main(state)
        qs2 = self._get_q_values_from_support(state)
        qs = (qs1 + qs2) / 2
        qs[action] = q
        self._batch_qs.append(qs)

        # now we train, and clear batches after
        if self._steps >= self._steps_to_train or episode_done:
            self._steps = 0
            bs = np.array(self._batch_states)
            bq = np.array(self._batch_qs)
            self._model.fit(bs, bq, epochs=self._epochs, verbose=0)

        # Update support model
        if episode_done:
            self._support_model.set_weights(self._model.get_weights())

    def update_v(self, state: Any, v: float, episode_done: bool, model_index: int = 0):
        raise Exception("Not implemented")

    def update_q_values(self, state: Any, values, episode_done: bool, model_index: int = 0):
        raise Exception("Not implemented")

    def get_max_a(self, state: Any, model_index: int = 0):
        raise Exception("Not implemented")

    def get_state_hash(self, state) -> Any:
        raise Exception("Not implemented")

    def get_q_values(self, state: Any, model_index: int = 0):
        return self._get_q_values_from_main(state)
        # if random.randrange(2) == 1:
        #    return self._get_q_values_from_main(state)
        # else:
        #    return self._get_q_values_from_support(state)

    def _get_q_values_from_main(self, state: Any):
        q_values = self._model(np.array([state]), training=False)
        return q_values[0].numpy()

    def _get_q_values_from_support(self, state: Any):
        q_values = self._support_model(np.array([state]), training=False)
        return q_values[0].numpy()
