from collections import deque
from gym import Env

from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.algorithms.MCACAlgorithm import UseCritic
from rl.algorithms.MCPGDACAlgorithm import MCPGDACAlgorithm
from rl.dyna.Dyna import Dyna
from rl.helpers.PlotHelper import PlotHelper
from rl.models.NNBasicModel import NNBasicModel
from rl.models.presets.Actors import Actors


# Monte Carlo Policy Gradient Discrete with Eligibility Traces
class CartPoleMCPGDLAgent(RDynaAgentBuilder):

    def __init__(self, model_path=None, load_model=False):
        self._model_path = model_path
        self._load_model = load_model
        stat_size = 200
        self._actor_losses = deque(maxlen=stat_size)
        self._rewards = deque(maxlen=stat_size)
        self._fig, self._axis = PlotHelper.build_subplots(2, 1000, 200, 8, 5)

    def build_agent(self, env: Env):
        actions = env.action_space.n
        alpha = 0.0001
        discount = 1
        epochs = 20
        input_shape = (4,)
        lambda_v = 0.97

        # So as we expected to use MC Algorithm, memory capacity should be equal to max episodes steps
        # it could be more, but it does not matter because memory will be cleared after every episode
        # if we have the number of steps in an episode more than capacity then it becomes TD(N) and you need to add
        # tail_method
        memory_capacity = 500

        algorithm = MCPGDACAlgorithm(discount=discount, memory_capacity=memory_capacity)
        algorithm.use_critic(UseCritic.No)

        actor = Actors(self._model_path, self._load_model).build_discrete_lambda(input_shape, actions, 200, alpha,
                                                                                 discount, lambda_v, epochs,
                                                                                 act=["relu", "relu"])

        actor.set_history_listener(history_listener=self._actor_history_listener)

        return Dyna(models=[actor], algorithm=algorithm)

    def _actor_history_listener(self, history, steps):
        NNBasicModel.add_history_loss_average(history, steps, self._actor_losses)

    def reward_listener(self, reward):
        self._rewards.append(reward)
        PlotHelper.plot(self._actor_losses, "Actor", self._axis[0], "r-")
        PlotHelper.plot(self._rewards, "Rewards", self._axis[1], "y-")
        PlotHelper.draw_all()
