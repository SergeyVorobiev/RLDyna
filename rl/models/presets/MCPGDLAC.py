from rl.helpers.ModelSaveLoadHelper import ModelSaveLoadHelper
from rl.models.NNBasicModel import NNBasicModel
from rl.models.losses.CustomLoss import CustomLoss
from rl.models.nnbuilders.CustomNetwork import CustomNetwork
from rl.models.nns.NNPGDLambda import NNPGDLambda
from rl.models.nns.NNTDLambda import NNTDLambda


# Monte carlo policy gradient discrete eligibility traces actor critic
class MCPGDLAC:

    def __init__(self, model_path, load_model):
        self._model_path = model_path
        self._load_model = load_model

    def build_u_critic(self, input_shape, alpha, discount, lambda_v, size, act, dropout, batch_normalization,
                       l1, l2, epochs, bias_init=None, cont_alpha=0.0,
                       model_index=1, run_eagerly=False, verbose=0) -> NNBasicModel:

        def critic_custom_model(inputs, x):
            return NNTDLambda(inputs, x, discount, lambda_v, cont_alpha)

        def build_nn():
            return CustomNetwork.u_critic(input_shape=input_shape, alpha=alpha, size=size,
                                          loss=CustomLoss.mc_loss_custom,
                                          act=act,
                                          out="linear",
                                          bias_init=bias_init,
                                          dropout=dropout,
                                          batch_normalization=batch_normalization,
                                          custom_model_build_func=critic_custom_model,
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

    def build_actor(self, input_shape, actions, size, alpha, discount, lambda_v, act, epochs,
                    dropout, batch_normalization, l1, l2, model_index=0, bias_init=None, run_eagerly=False, verbose=0):

        def actor_custom_model(inputs, x):
            return NNPGDLambda(inputs, x, discount, lambda_v)

        def build_nn():
            return CustomNetwork.two_hidden(input_shape=input_shape, output_n=actions,
                                            alpha=alpha,
                                            size=size,
                                            loss=CustomLoss.mcpgd_traces,
                                            act=act,
                                            bias_init=bias_init,
                                            l1=l1,
                                            l2=l2,
                                            custom_model_build_func=actor_custom_model,
                                            dropout=dropout,
                                            run_eagerly=run_eagerly,
                                            batch_normalization=batch_normalization,
                                            out="softmax")

        def custom_load_actor_model_func_weights(path):
            return ModelSaveLoadHelper.load_weights_h5(path, build_nn)

        return NNBasicModel(n_actions=actions, nn_build_function=build_nn, model_path=self._model_path,
                            load_model=self._load_model,
                            custom_load_model_func=custom_load_actor_model_func_weights,
                            custom_save_model_func=ModelSaveLoadHelper.save_weights_h5,
                            epochs=epochs, verbose=verbose, model_index=model_index)
