import keras

from rl.helpers.ModelSaveLoadHelper import ModelSaveLoadHelper
from rl.models.NNBasicModel import NNBasicModel
from rl.models.nnbuilders.CustomNetwork import CustomNetwork


# Policy Gradient Discrete and Cont + TD
class MCPGDAC:

    def __init__(self):
        self._custom_critic_loss = None
        self._custom_actor_loss = None
        self._actor_load_model = self._custom_load_actor_model_func
        self._critic_load_model = self._custom_load_critic_model_func
        self._actor_save_model = ModelSaveLoadHelper.simple_save
        self._critic_save_model = ModelSaveLoadHelper.simple_save
        self._critic_build_nn = None
        self._actor_build_nn = None

    # Actor MCPGDiscrete, critic MCU
    def build_acd(self, input_shape, n_actions, actor_alpha, critic_alpha, actor_layer_size, critic_layer_size,
                  actor_activation, critic_activation, actor_epochs, critic_epochs, model_path, load_model, actor_loss,
                  critic_loss, actor_dropout=0, critic_dropout=0, actor_batch_normalization=False,
                  critic_batch_normalization=False, actor_custom_model=None, critic_custom_model=None,
                  run_eagerly=False,
                  save_load_only_weights=False, verbose=0, al1=0, al2=0, cl1=0, cl2=0):
        self._custom_critic_loss = critic_loss
        self._custom_actor_loss = actor_loss

        if save_load_only_weights:
            self._critic_load_model = self._custom_load_critic_model_func_weights
            self._actor_load_model = self._custom_load_actor_model_func_weights
            self._actor_save_model = ModelSaveLoadHelper.save_weights_h5
            self._critic_save_model = ModelSaveLoadHelper.save_weights_h5
        actor = self._get_actor_model(input_shape, n_actions, actor_layer_size, actor_alpha,
                                      actor_activation, model_path, load_model, actor_epochs,
                                      actor_dropout, actor_batch_normalization, actor_loss,
                                      actor_custom_model, run_eagerly, verbose, al1, al2)
        critic = self._get_critic_model(input_shape, critic_layer_size, critic_alpha,
                                        critic_activation, model_path, load_model, critic_epochs,
                                        critic_dropout, critic_batch_normalization, critic_loss,
                                        critic_custom_model, run_eagerly, verbose, cl1, cl2)
        return actor, critic

    def _get_actor_model(self, input_shape, actions, size, alpha, act, model_path, load_model, epochs, dropout,
                         batch_normalization, loss, custom_model, run_eagerly, verbose, l1, l2):
        self._actor_build_nn = lambda: CustomNetwork.two_hidden(input_shape=input_shape, output_n=actions,
                                                                alpha=alpha,
                                                                size=size,
                                                                loss=loss,
                                                                act=act,
                                                                l1=l1,
                                                                l2=l2,
                                                                custom_model_build_func=custom_model,
                                                                dropout=dropout,
                                                                run_eagerly=run_eagerly,
                                                                batch_normalization=batch_normalization,
                                                                out="softmax")

        return NNBasicModel(n_actions=actions, nn_build_function=self._actor_build_nn, model_path=model_path,
                            load_model=load_model,
                            custom_load_model_func=self._actor_load_model,
                            custom_save_model_func=self._actor_save_model,
                            epochs=epochs, verbose=verbose)

    def _get_critic_model(self, input_shape, size, alpha, act, model_path, load_model, epochs, dropout,
                          batch_normalization, loss, custom_model, run_eagerly, verbose, l1, l2):
        self._critic_build_nn = lambda: CustomNetwork.u_critic(input_shape=input_shape,
                                                               alpha=alpha,
                                                               size=size,
                                                               dropout=dropout,
                                                               batch_normalization=batch_normalization,
                                                               custom_model_build_func=custom_model,
                                                               l1=l1,
                                                               l2=l2,
                                                               run_eagerly=run_eagerly,
                                                               loss=loss, act=act,
                                                               out="linear")

        # n_actions it is just expected G for state as output in this case
        return NNBasicModel(n_actions=1, nn_build_function=self._critic_build_nn, model_path=model_path,
                            custom_load_model_func=self._critic_load_model,
                            custom_save_model_func=self._critic_save_model,
                            load_model=load_model, epochs=epochs, model_index=1, verbose=verbose)

    def _custom_load_actor_model_func(self, path):
        try:
            return keras.models.load_model(path, custom_objects={
                self._custom_actor_loss.__name__: self._custom_actor_loss})
        except IOError as e:
            return None

    def _custom_load_critic_model_func(self, path):
        try:
            return keras.models.load_model(path, custom_objects={
                self._custom_critic_loss.__name__: self._custom_critic_loss})
        except IOError as e:
            return None

    def _custom_load_critic_model_func_weights(self, path):
        return ModelSaveLoadHelper.load_weights_h5(path, self._critic_build_nn)

    def _custom_load_actor_model_func_weights(self, path):
        return ModelSaveLoadHelper.load_weights_h5(path, self._actor_build_nn)
