from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.algorithms.MCPGTDAlgorithm import MCPGTDAlgorithm
from rl.dyna.Dyna import Dyna
from rl.environments.fl.BasicGridEnv import BasicGridEnv
from rl.models.nnbuilders.MCPGTDActorCriticBuilder import MCPGTDActorCriticDiscreteBuilder


# Monte Carlo Policy Gradient + TD Actor-Critic
class LunarLanderMCPGTDActorCriticAgent(RDynaAgentBuilder):

    def __init__(self, model_path=None, load_model=False):
        self._model_path = model_path
        self._load_model = load_model

    def build_agent(self, env: BasicGridEnv):
        actions = env.action_space.n
        actor_alpha = 0.00001
        critic_alpha = 0.00001
        discount = 1

        # In Monte Carlo algorithm we perform the computation only after the episode ends.
        # Be sure that memory_capacity is enough to accomplish the episode of the task or the algorithm
        # will calculate the episode based on not full steps that likely does not lead to convergence.
        algorithm = MCPGTDAlgorithm(discount=discount, memory_capacity=1000)
        actor, critic = MCPGTDActorCriticDiscreteBuilder.build(input_shape=(8,),
                                                               n_actions=actions,
                                                               actor_alpha=actor_alpha,
                                                               critic_alpha=critic_alpha,
                                                               actor_layer_size=500,
                                                               critic_layer_size=500,
                                                               actor_activation="tanh",
                                                               critic_activation="relu",
                                                               epochs=10,
                                                               model_path=self._model_path,
                                                               load_model=self._load_model)
        return Dyna(models=[actor, critic], algorithm=algorithm)
