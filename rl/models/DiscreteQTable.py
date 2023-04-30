import os
from typing import Any

import numpy as np

from rl.models.RModel import RModel


# This table is useful when you have evenly distributed values that should be placed in n-dim discrete table
class DiscreteQTable(RModel):

    # dim_min_max - [[-1, 1][-2, 2][-1, 1][-2, 2]] means 4 dimension table with min-max pairs for each dimension
    # quant_size - for each dimension will evenly assign values to each cell 5 for [-1, 1] means -1, -0.5, 0, 0.5, 1
    def __init__(self, dim_min_max: [], quant_size: int, n_actions, model_path=None, load_model=False, model_index=0):
        super().__init__(n_actions)
        dimensions = dim_min_max.__len__()
        self._quants = []
        if model_path is not None:
            model_path = model_path + "_" + str(model_index)
        self._model_path = model_path
        self._load_model = load_model
        self._model_index = model_index
        for min_max in dim_min_max:
            self._quants.append(np.linspace(min_max[0], min_max[1], quant_size))
        if self._model_path is not None and load_model:
            try:
                self._q_table = np.load(self._model_path, allow_pickle=True)
                print("Model is loaded: " + self._model_path)
            except IOError as e:
                print("Model is not found: " + self._model_path)
                self._q_table = np.zeros(shape=([quant_size] * dimensions + [n_actions]))
        else:
            self._q_table = np.zeros(shape=([quant_size] * dimensions + [n_actions]))

    def update(self, data: Any, batch_size=32):
        raise NotImplementedError

    # returns table indexes
    def digitize_state(self, state: Any):
        digitized = []
        for i in range(len(state)):
            digitized.append(np.digitize(state[i], self._quants[i]) - 1)
        return tuple(digitized)

    def get_q_values(self, state: Any, model_index: int = 0):
        return self._q_table[state]

    def get_max_q(self, state: Any, model_index: int = 0):
        return max(self.get_q_values(state))

    def get_q(self, state, action: int, model_index: int = 0):
        return self.get_q_values(state)[action]

    def get_v(self, state: Any, model_index: int = 0):
        raise Exception("Not implemented")

    def update_q(self, state: Any, action: int, q: float, episode_done: bool, model_index: int = 0):
        self.get_q_values(state)[action] = q

    def update_v(self, state: Any, v: float, episode_done: bool, model_index: int = 0):
        raise Exception("Not implemented")

    def update_q_values(self, state: Any, values, episode_done: bool, model_index: int = 0):
        pass

    def get_max_a(self, state: Any, model_index: int = 0):
        np.argmax(self.get_q_values(state))

    def get_state_hash(self, state) -> Any:
        hash(state.data.tobytes())

    def save(self, path=None) -> (bool, str):
        if path is not None:
            path = path + "_" + str(self._model_index)
            self._save_by_path(path)
            return True, path
        elif self._model_path is not None:
            self._save_by_path(self._model_path)
            return True, self._model_path
        return False, None

    def _save_by_path(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            np.save(f, self._q_table)

    def get_action_values(self, state: Any, model_index: int = 0):
        raise NotImplementedError
