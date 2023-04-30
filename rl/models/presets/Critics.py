from rl.models.presets.MCPGDLAC import MCPGDLAC


class Critics:

    def __init__(self, model_path, load_model):
        self._model_path = model_path
        self._load_model = load_model

    def build_discrete_lambda(self, input_shape, alpha, size, discount, lambda_v, cont_alpha, epochs, act=None):
        if act is None:
            act = ["tanh", "relu"]
        builder = MCPGDLAC(self._model_path, self._load_model)
        return builder.build_u_critic(input_shape=input_shape,
                                      alpha=alpha,
                                      discount=discount,
                                      lambda_v=lambda_v,
                                      bias_init="glorot_uniform",
                                      size=size,
                                      act=act,
                                      dropout=0.000,
                                      batch_normalization=False,
                                      l1=0.000,
                                      l2=0.000,
                                      epochs=epochs,
                                      cont_alpha=cont_alpha,
                                      model_index=1,
                                      run_eagerly=False,
                                      verbose=0)
