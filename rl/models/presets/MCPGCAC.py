import keras

from rl.helpers.ModelSaveLoadHelper import ModelSaveLoadHelper
from rl.models.NNBasicModel import NNBasicModel
from rl.models.losses.CustomLoss import CustomLoss
from rl.models.nnbuilders.CustomNetwork import CustomNetwork


# Monte carlo policy gradient continuous actor critic
from rl.models.nns.NNPGC import NNPGC


class MCPGCAC:

    def __init__(self, model_path, load_model):
        self._model_path = model_path
        self._load_model = load_model

    def build_gauss_actor(self, input_shape, actions, alpha, size, act_mean, act_deviation,
                          batch_normalization, epochs, dropout=0.0, l1=0.0, l2=0.0, model_index=0, run_eagerly=False,
                          bias_init=None, verbose=0) -> NNBasicModel:

        def build_nn():
            return CustomNetwork.build_distribution(input_shape=input_shape,
                                                    output_n=actions,
                                                    alpha=alpha,
                                                    size=size,
                                                    act_mean=act_mean,
                                                    bias_init=bias_init,
                                                    l1=l1,
                                                    l2=l2,
                                                    dropout=dropout,
                                                    loss=None,
                                                    act_deviation=act_deviation,
                                                    batch_normalization=batch_normalization,
                                                    custom_model_build_func=NNPGC,
                                                    run_eagerly=run_eagerly)

        def custom_load_model_func_weights(path):
            return ModelSaveLoadHelper.load_weights_h5(path, build_nn)

        return NNBasicModel(n_actions=0, nn_build_function=build_nn, model_path=self._model_path,
                            load_model=self._load_model,
                            custom_load_model_func=custom_load_model_func_weights,
                            custom_save_model_func=ModelSaveLoadHelper.save_weights_h5,
                            model_index=model_index,
                            epochs=epochs,
                            verbose=verbose)

    @staticmethod
    def _custom_load_model_func(path):
        try:
            return keras.models.load_model(path, custom_objects={
                CustomLoss.custom_loss_gaussian.__name__: CustomLoss.custom_loss_gaussian})
        except IOError as e:
            return None
