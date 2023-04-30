from abc import ABC
from typing import Any
from rl.models.NNBasicModel import NNBasicModel
import tensorflow as tf


class SARSAModelForNN(NNBasicModel, ABC):

    # nn_build_function - in case if the model can not be loaded
    def __init__(self, n_actions, nn_build_function, epochs=1, model_path=None, load_model=None, model_index=0,
                 model_signatures=None, custom_save_model_func=None,
                 custom_load_model_func=None, history_listener=None, verbose=0):
        super().__init__(n_actions, nn_build_function, epochs, model_path, load_model, model_index,
                         model_signatures, custom_save_model_func, custom_load_model_func, history_listener, verbose)

    def get_q_values(self, state: Any, model_index: int = 0):
        q_values = self._model(tf.convert_to_tensor([state]), training=False)
        return q_values[0].numpy()

    def get_q(self, state, action: int, model_index: int = 0):
        q_values = self.get_q_values(state)
        return q_values[action]
