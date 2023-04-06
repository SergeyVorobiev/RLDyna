import keras

from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.algorithms.MCPGAverBaselineAlgorithm import MCPGAverBaselineAlgorithm
from rl.dyna.Dyna import Dyna
from rl.environments.fl.BasicGridEnv import BasicGridEnv
from rl.models.losses.CustomLoss import CustomLoss
from rl.models.MCPGModel import MCPGModel
from rl.models.nnbuilders.CustomNetwork import CustomNetwork


# Monte Carlo Policy Gradient control
class CartPoleMCPGAgent(RDynaAgentBuilder):

    def __init__(self, model_path=None, load_model=False):
        self._model_path = model_path
        self._load_model = load_model

    def build_agent(self, env: BasicGridEnv):
        actions = env.action_space.n
        alpha = 0.001
        discount = 0.99

        build_nn = lambda: CustomNetwork.build_linear(input_shape=(4,), output_n=actions, alpha=alpha, size=100,
                                                      loss=CustomLoss.mc_policy_gradient, act="relu", out="softmax")

        # model_signatures = lambda model: MCPGNNDiscrete.get_signatures((1, 4), model)

        algorithm = MCPGAverBaselineAlgorithm(alpha=alpha, discount=discount, memory_capacity=550, use_baseline=True)
        models = [MCPGModel(n_actions=actions, nn_build_function=build_nn, model_path=self._model_path,
                            load_model=self._load_model,
                            custom_load_model_func=CartPoleMCPGAgent.custom_load_model_func,
                            custom_save_model_func=CartPoleMCPGAgent._custom_save_model_func,
                            epochs=5)]
        return Dyna(models=models, algorithm=algorithm)

    @staticmethod
    def custom_load_model_func(path):
        try:
            return keras.models.load_model(path, custom_objects={
                CustomLoss.mc_policy_gradient.__name__: CustomLoss.mc_policy_gradient})
        except IOError as e:
            return None

    @staticmethod
    def _custom_save_model_func(model, path):
        model.save(path)
