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
class LunarLanderMCPGDLAgent(RDynaAgentBuilder):

    def __init__(self, model_path=None, load_model=False):
        self._model_path = model_path
        self._load_model = load_model

        stat_size = 200
        self._actor_losses = deque(maxlen=stat_size)
        self._rewards = deque(maxlen=stat_size)
        self._critic_losses = deque(maxlen=stat_size)
        self._action1 = deque(maxlen=stat_size)
        self._action2 = deque(maxlen=stat_size)
        self._action3 = deque(maxlen=stat_size)
        self._action4 = deque(maxlen=stat_size)
        self._fig, self._axis = PlotHelper.build_subplots(6, 1000, 50)

    def build_agent(self, env: Env):
        input_shape = (8,)
        actions = env.action_space.n
        alpha = 0.00001
        lambda_v = 0.0
        discount = 1
        epochs = 20

        # If memory capacity is less than the size of the episode it will be considered as TD(N)
        algorithm = MCPGDACAlgorithm(discount=discount, memory_capacity=1000, actions_listener=self._actions_listener)
        algorithm.use_critic(UseCritic.No)

        actors = Actors(self._model_path, self._load_model)
        # actor = actors.build_discrete(input_shape, actions, 400, alpha, discount, epochs)
        actor = actors.build_discrete_lambda(input_shape, actions, 400, alpha, discount, lambda_v, epochs)
        actor.set_history_listener(history_listener=self._actor_history_listener)

        return Dyna(models=[actor], algorithm=algorithm)

    def reward_listener(self, reward):
        self._rewards.append(reward)
        PlotHelper.plot(self._actor_losses, "Actor", self._axis[0], "r-")
        PlotHelper.plot(self._rewards, "Reward", self._axis[1], "y-")
        PlotHelper.plot(self._action1, "Action1", self._axis[2], "b-")
        PlotHelper.plot(self._action2, "Action2", self._axis[3], "b-")
        PlotHelper.plot(self._action3, "Action3", self._axis[4], "b-")
        PlotHelper.plot(self._action4, "Action4", self._axis[5], "b-")
        PlotHelper.draw_all()

    def _actions_listener(self, actions):
        self._action1.append(actions[0])
        self._action2.append(actions[1])
        self._action3.append(actions[2])
        self._action4.append(actions[3])

    def _actor_history_listener(self, history, steps):
        NNBasicModel.add_history_loss_average(history, steps, self._actor_losses)
