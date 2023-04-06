from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import tensorflow as tf

# pip install tensorflow-probability
from tensorflow_probability.python.distributions import Categorical

from rl.models.PolicyGradientAbsModel import PolicyGradientAbsModel


class NNBasicModel(PolicyGradientAbsModel, ABC):

    # nn_build_function - in case if the model can not be loaded
    def __init__(self, n_actions, nn_build_function, epochs=1, model_path=None, load_model=None, model_index=0,
                 model_signatures=None, custom_save_model_func=None,
                 custom_load_model_func=None):
        super().__init__(n_actions)
        self._nn_build_func = nn_build_function
        if model_path is not None:
            model_path = model_path + "_" + str(model_index)
        self._model_path = model_path
        self._model_index = model_index
        self._load_model = load_model
        self._epochs = epochs
        self._custom_save_model_func = custom_save_model_func
        self._custom_load_model_func = custom_load_model_func
        self._model_signatures = model_signatures
        self._model = self._load()

    def save(self, path=None) -> (bool, str):
        if path is not None:
            path = path + "_" + str(self._model_index)
            if self._custom_save_model_func is not None:
                self._custom_save_model_func(self._model, path)
            else:
                self._save_model(path)
            return True, path
        elif self._model_path is not None:
            if self._custom_save_model_func is not None:
                self._custom_save_model_func(self._model, self._model_path)
            else:
                self._save_model(self._model_path)
            return True, self._model_path
        return False, None

    def _save_model(self, path):
        if self._model_signatures is not None:
            tf.saved_model.save(self._model, path, signatures=self._model_signatures(self._model))
        else:
            tf.saved_model.save(self._model, path)

    @abstractmethod
    def update(self, data: Any):
        ...

    def get_max_a(self, state: Any, model_index: int = 0):
        action_probs = self._model(np.array([state]), training=False)
        dist = Categorical(probs=action_probs, dtype=tf.float32)
        return int(dist.sample())  # Get action based on the probability

    def get_a_distribution(self, state: Any, model_index: int = 0):
        return np.array(self._model(np.array([state]), training=False)[0])

    def get_state_hash(self, state) -> Any:
        return hash(state.data.tobytes())

    def _load(self):
        if self._model_path is not None and self._load_model:
            try:
                if self._custom_load_model_func is not None:
                    model = self._custom_load_model_func(self._model_path)
                else:
                    model = tf.saved_model.load(self._model_path)
                if model is None:
                    print("Can not load model (None): " + self._model_path)
                    model = self._nn_build_func()
                else:
                    print("Model is loaded: " + self._model_path)
                return model
            except IOError as e:
                print("Model is not found: " + self._model_path)
                return self._nn_build_func()
        else:
            return self._nn_build_func()
