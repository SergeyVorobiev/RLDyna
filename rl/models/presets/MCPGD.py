import keras
from rl.models.NNBasicModel import NNBasicModel
from rl.models.losses.CustomLoss import CustomLoss
from rl.models.nnbuilders.CustomNetwork import CustomNetwork


class MCPGD:

    def __init__(self, model_path, load_model, alpha, discount):
        self._alpha = alpha
        self._discount = discount
        self._model_path = model_path
        self._load_model = load_model

    def build(self, input_shape, actions, size, act, dropout, batch_normalization, l1, l2, epochs,
              bias_init=None) -> NNBasicModel:

        def build_nn():
            return CustomNetwork.two_hidden(input_shape=input_shape, output_n=actions, alpha=self._alpha, size=size,
                                            loss=CustomLoss.mcpgd,
                                            act=act,
                                            bias_init=bias_init,
                                            out="softmax",
                                            dropout=dropout,
                                            batch_normalization=batch_normalization,
                                            l1=l1,
                                            l2=l2)

        return NNBasicModel(n_actions=actions, nn_build_function=build_nn, model_path=self._model_path,
                            load_model=self._load_model,
                            custom_load_model_func=self._custom_load_model_func,
                            custom_save_model_func=self._custom_save_model_func,
                            epochs=epochs)

    @staticmethod
    def _custom_load_model_func(path):
        try:
            return keras.models.load_model(path, custom_objects={
                CustomLoss.mcpgd.__name__: CustomLoss.mcpgd})
        except IOError as e:
            return None

    @staticmethod
    def _custom_save_model_func(model, path):
        model.save(path)
