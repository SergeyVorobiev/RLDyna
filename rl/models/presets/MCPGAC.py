from rl.helpers.ModelSaveLoadHelper import ModelSaveLoadHelper
from rl.models.NNBasicModel import NNBasicModel
from rl.models.losses.CustomLoss import CustomLoss
from rl.models.nnbuilders.CustomNetwork import CustomNetwork
from rl.models.nns.NNTD import NNTD


# Monte carlo policy gradient actor critic
class MCPGAC:

    def __init__(self, model_path, load_model):
        self._model_path = model_path
        self._load_model = load_model

    def build_u_critic(self, input_shape, alpha, size, act, dropout, batch_normalization, epochs,
                       bias_init=None,
                       l1=0.0, l2=0.0, model_index=1, run_eagerly=False, verbose=0) -> NNBasicModel:

        def build_nn():
            return CustomNetwork.u_critic(input_shape=input_shape, alpha=alpha, size=size,
                                          loss=CustomLoss.mc_loss_custom,
                                          act=act,
                                          bias_init=bias_init,
                                          out="linear",
                                          dropout=dropout,
                                          batch_normalization=batch_normalization,
                                          custom_model_build_func=NNTD,
                                          l1=l1,
                                          l2=l2,
                                          run_eagerly=run_eagerly)

        def custom_load_critic_model_func_weights(path):
            return ModelSaveLoadHelper.load_weights_h5(path, build_nn)

        return NNBasicModel(n_actions=0, nn_build_function=build_nn, model_path=self._model_path,
                            load_model=self._load_model,
                            custom_load_model_func=custom_load_critic_model_func_weights,
                            custom_save_model_func=ModelSaveLoadHelper.save_weights_h5,
                            model_index=model_index,
                            epochs=epochs,
                            verbose=verbose)
