from abc import ABC
from typing import Any

import tensorflow as tf
from rl.models.NNBasicModel import NNBasicModel


class MCTDModel(NNBasicModel, ABC):

    # nn_build_function - in case if the model can not be loaded
    def __init__(self, n_actions, nn_build_function, epochs=1, model_path=None, load_model=None, model_index=0,
                 model_signatures=None, custom_save_model_func=None,
                 custom_load_model_func=None):
        super().__init__(n_actions, nn_build_function, epochs, model_path, load_model, model_index,
                         model_signatures, custom_save_model_func, custom_load_model_func)

    def update(self, data: Any):
        x = tf.convert_to_tensor(data[0])
        y = tf.convert_to_tensor(data[1])

        # Losses can be gotten from the history as a result of fit
        self._model.fit(x=x, y=y, epochs=self._epochs, verbose=0)

