from rl.models.presets.MCPGCAC import MCPGCAC
from rl.models.presets.MCPGD import MCPGD
from rl.models.presets.MCPGDLAC import MCPGDLAC


class Actors:

    def __init__(self, model_path, load_model):
        self._model_path = model_path
        self._load_model = load_model

    def build_discrete_lambda(self, input_shape, actions, size, alpha, discount, lambda_v, epochs, act=None):
        if act is None:
            act = ["tanh", "relu"]
        return MCPGDLAC(self._model_path, self._load_model).build_actor(input_shape=input_shape,
                                                                        actions=actions,
                                                                        size=size,
                                                                        alpha=alpha,
                                                                        discount=discount,
                                                                        lambda_v=lambda_v,
                                                                        act=act,
                                                                        bias_init="glorot_uniform",
                                                                        epochs=epochs,
                                                                        dropout=0.000,
                                                                        batch_normalization=False,
                                                                        l1=0.000,
                                                                        l2=0.000,
                                                                        model_index=0,
                                                                        run_eagerly=False,
                                                                        verbose=0)

    def build_discrete(self, input_shape, actions, size, alpha, discount, epochs, act=None):
        if act is None:
            act = ["tanh", "relu"]
        return MCPGD(self._model_path, self._load_model, alpha, discount).build(input_shape=input_shape,
                                                                                actions=actions,
                                                                                size=size,
                                                                                act=act,
                                                                                dropout=0.000,
                                                                                batch_normalization=False,
                                                                                bias_init="glorot_uniform",
                                                                                l1=0.0001,
                                                                                l2=0.0001,
                                                                                epochs=epochs)

    def build_gauss(self, input_shape, actions, size, alpha, epochs, act=None):
        if act is None:
            act = ["tanh", "elu"]
        return MCPGCAC(self._model_path, self._load_model).build_gauss_actor(input_shape=input_shape,
                                                                             actions=actions,
                                                                             alpha=alpha,
                                                                             size=size,
                                                                             act_mean=act,
                                                                             act_deviation=act,
                                                                             batch_normalization=False,
                                                                             dropout=0.00,
                                                                             l1=0.000,
                                                                             l2=0.000,
                                                                             bias_init="glorot_uniform",
                                                                             epochs=epochs,
                                                                             model_index=0,
                                                                             run_eagerly=False,
                                                                             verbose=0)
