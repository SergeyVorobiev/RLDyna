from collections import deque

from gym import Env

from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.algorithms.MCACAlgorithm import UseCritic
from rl.algorithms.MCPGCACAlgorithm import MCPGCACAlgorithm
from rl.dyna.Dyna import Dyna
from rl.helpers.PlotHelper import PlotHelper
from rl.models.NNBasicModel import NNBasicModel
from rl.models.presets.Actors import Actors
from rl.tasks.mountaincar.MountainCarLinearStatePrepare import MountainCarLinearStatePrepare


# Monte Carlo Policy Gradient Continuous
class LunarLanderMCPGCAgent(RDynaAgentBuilder):

    def __init__(self, model_path=None, load_model=False):
        self._model_path = model_path
        self._load_model = load_model
        self._size = 1000

        self._actor_losses = deque(maxlen=self._size)
        self._critic_losses = deque(maxlen=self._size)
        self._deviations = deque(maxlen=self._size)
        self._means = deque(maxlen=self._size)
        self._actions = deque(maxlen=self._size)
        self._rewards = deque(maxlen=self._size)
        self._means2 = deque(maxlen=self._size)
        self._actions2 = deque(maxlen=self._size)
        self._deviations2 = deque(maxlen=self._size)

        self._fig, self._axis = PlotHelper.build_subplots(8, 1000, 20)
        self._state_prepare = MountainCarLinearStatePrepare()
        self._actor = None

    def build_agent(self, env: Env):
        shape = (8,)
        alpha = 0.000001
        discount = 1
        epochs = 10
        actions = 2

        memory_capacity = 1000
        algorithm = MCPGCACAlgorithm(discount=discount, memory_capacity=memory_capacity,
                                     gauss_listener=self._gauss_listener)
        algorithm.use_critic(UseCritic.No)
        self._actor = Actors(self._model_path, self._load_model).build_gauss(shape, actions, 400, alpha, epochs)

        models = [self._actor]

        self._actor.set_history_listener(history_listener=self._actor_history_listener)

        return Dyna(models=models, algorithm=algorithm)

    def _actor_history_listener(self, history, steps):
        NNBasicModel.add_history_loss_average(history, steps, self._actor_losses)

    def reward_listener(self, reward):
        self._rewards.append(reward)
        PlotHelper.plot(self._actor_losses, "Actor", self._axis[0], "r-")
        PlotHelper.plot(self._rewards, "Reward", self._axis[1], "y-")
        PlotHelper.plot(self._deviations, "Deviation1", self._axis[2], "g-")
        PlotHelper.plot(self._means, "Mean1", self._axis[3], "-")
        PlotHelper.plot(self._actions, "Action1", self._axis[4], "-")
        PlotHelper.plot(self._deviations2, "Deviation2", self._axis[5], "g-")
        PlotHelper.plot(self._means2, "Mean2", self._axis[6], "-")
        PlotHelper.plot(self._actions2, "Action2", self._axis[7], "-")
        PlotHelper.draw_all()

    def _gauss_listener(self, mean, deviation, action):
        self._means.append(mean[0])
        self._deviations.append(deviation[0])
        self._actions.append(action[0])
        self._means2.append(mean[1])
        self._deviations2.append(deviation[1])
        self._actions2.append(action[1])
