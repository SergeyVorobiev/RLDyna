from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.algorithms.MCPGAverBaselineAlgorithm import MCPGAverBaselineAlgorithm
from rl.dyna.Dyna import Dyna
from rl.environments.fl.BasicGridEnv import BasicGridEnv, StateType
from rl.models.MCPGModel import MCPGModel
from rl.models.nnbuilders.ShelterNetwork import ShelterNetwork
from rl.tasks.shelter.ShelterStateNNPrepare import ShelterStateNNPrepare


# Monte Carlo Policy Gradient control
class MCPGAgent(RDynaAgentBuilder):

    def __init__(self, model_path=None, load_model=False):
        self._model_path = model_path
        self._load_model = load_model

    def build_agent(self, env: BasicGridEnv):
        env.set_state_type(StateType.all_map)
        actions = env.action_space.n
        alpha = 0.001
        discount = 1
        build_nn = lambda: ShelterNetwork.build_model(input_shape=(8, 7, 1), alpha=alpha)

        # Iterative algorithm
        algorithm = MCPGAverBaselineAlgorithm(alpha=alpha, discount=discount, memory_capacity=100000)
        models = [MCPGModel(n_actions=actions, nn_build_function=build_nn, model_path=self._model_path,
                            load_model=self._load_model)]

        return Dyna(models=models, algorithm=algorithm, state_prepare=ShelterStateNNPrepare())
