import keras

from rl.models.MCPGModel import MCPGModel
from rl.models.MCTDModel import MCTDModel
from rl.models.losses.CustomLoss import CustomLoss
from rl.models.nnbuilders.CustomNetwork import CustomNetwork


class MCPGTDActorCriticDiscreteBuilder:

    def __init__(self):
        pass

    @staticmethod
    def build(input_shape, n_actions, actor_alpha, critic_alpha, actor_layer_size, critic_layer_size,
              actor_activation, critic_activation, epochs, model_path, load_model):
        actor = MCPGTDActorCriticDiscreteBuilder._get_actor_model(input_shape, n_actions, actor_layer_size, actor_alpha,
                                                                  actor_activation, model_path, load_model, epochs)
        critic = MCPGTDActorCriticDiscreteBuilder._get_critic_model(input_shape, critic_layer_size, critic_alpha,
                                                                    critic_activation, model_path, load_model, epochs)
        return actor, critic

    @staticmethod
    def _get_actor_model(input_shape, actions, size, alpha, act, model_path, load_model, epochs):
        build_nn = lambda: CustomNetwork.build_linear(input_shape=input_shape, output_n=actions, alpha=alpha,
                                                      size=size,
                                                      loss=CustomLoss.mc_policy_gradient, act=act,
                                                      out="softmax")

        return MCPGModel(n_actions=actions, nn_build_function=build_nn, model_path=model_path,
                         load_model=load_model,
                         custom_load_model_func=MCPGTDActorCriticDiscreteBuilder._custom_load_actor_model_func,
                         custom_save_model_func=MCPGTDActorCriticDiscreteBuilder._custom_save_model_func,
                         epochs=epochs)

    @staticmethod
    def _get_critic_model(input_shape, size, alpha, act, model_path, load_model, epochs):
        build_nn = lambda: CustomNetwork.build_linear(input_shape=input_shape, output_n=1, alpha=alpha,
                                                      size=size,
                                                      loss=CustomLoss.td_loss, act=act,
                                                      out="linear")

        # n_actions it is just expected G for state as output in this case
        return MCTDModel(n_actions=1, nn_build_function=build_nn, model_path=model_path,
                         custom_load_model_func=MCPGTDActorCriticDiscreteBuilder._custom_load_critic_model_func,
                         custom_save_model_func=MCPGTDActorCriticDiscreteBuilder._custom_save_model_func,
                         load_model=load_model, epochs=epochs, model_index=1)

    @staticmethod
    def _custom_load_actor_model_func(path):
        try:
            return keras.models.load_model(path, custom_objects={
                CustomLoss.mc_policy_gradient.__name__: CustomLoss.mc_policy_gradient})
        except IOError as e:
            return None

    @staticmethod
    def _custom_load_critic_model_func(path):
        try:
            return keras.models.load_model(path, custom_objects={
                CustomLoss.td_loss.__name__: CustomLoss.td_loss})
        except IOError as e:
            return None

    @staticmethod
    def _custom_save_model_func(model, path):
        model.save(path)