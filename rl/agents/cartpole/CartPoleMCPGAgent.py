from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.algorithms.MCPGAverBaseline import MCPGAverBaseline
from rl.dyna.Dyna import Dyna
from rl.environments.fl.BasicGridEnv import BasicGridEnv
from rl.models.nns.MCPGNNDiscrete import MCPGNNDiscrete
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
                                                      custom_model=MCPGNNDiscrete)

        model_signatures = lambda model: MCPGNNDiscrete.get_signatures((1, 4), model)

        algorithm = MCPGAverBaseline(alpha=alpha, discount=discount, memory_capacity=550, use_baseline=True)
        models = [MCPGModel(n_actions=actions, nn_build_function=build_nn, model_path=self._model_path,
                            load_model=self._load_model, model_signatures=model_signatures)]
        return Dyna(models=models, algorithm=algorithm)
