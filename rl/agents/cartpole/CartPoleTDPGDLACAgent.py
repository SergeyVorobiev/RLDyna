from collections import deque
from gym import Env

from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.algorithms.MCACAlgorithm import UseCritic
from rl.algorithms.MCPGDACAlgorithm import MCPGDACAlgorithm
from rl.dyna.Dyna import Dyna
from rl.helpers.PlotHelper import PlotHelper
from rl.models.NNBasicModel import NNBasicModel
from rl.models.presets.Actors import Actors
from rl.models.presets.Critics import Critics


# N steps Temporal Difference Policy Gradient discrete eligibility traces + Baseline Critic with eligibility traces
class CartPoleTDPGDLACAgent(RDynaAgentBuilder):

    def __init__(self, model_path=None, load_model=False):
        self._model_path = model_path
        self._load_model = load_model
        stat_size = 200
        self._actor_losses = deque(maxlen=stat_size)
        self._critic_losses = deque(maxlen=stat_size)
        self._rewards = deque(maxlen=stat_size)
        self._fig, self._axis = PlotHelper.build_subplots(3, 1000, 200, 8, 5)

    def build_agent(self, env: Env):
        actions = env.action_space.n
        actor_alpha = 0.00001
        critic_alpha = 0.0001
        cont_alpha = 0.0001
        lambda_v = 0.97
        discount = 1
        epochs = 10
        input_shape = (4,)

        # Useless for MC because memory will be cleared automatically after every episode
        # Speeds up TD(N), will send batch to learn in order 1-5, 5-10 instead of 1-5, 2-6, 3-7...
        clear_memory_every_n_steps = True

        # 500 will mean full episode MC, now its TD(5)
        memory_capacity = 5

        # It is useful if you have memory_capacity less than the episode steps in this case MC will automatically be
        # considered as TD(N) but then you need to add Unext, this method describes how to calculate the value of
        # Unext
        def get_tail(models, n_state):
            return float(models[1].predict([n_state]))

        algorithm = MCPGDACAlgorithm(discount=discount, memory_capacity=memory_capacity, tail_method=get_tail,
                                     clear_memory_every_n_steps=clear_memory_every_n_steps)

        # Disable critic if you use full episode, Enabled by default.
        algorithm.use_critic(UseCritic.Yes)

        actor = Actors(self._model_path, self._load_model).build_discrete_lambda(input_shape, actions, 200, actor_alpha,
                                                                                 discount, lambda_v, epochs)
        critic = Critics(self._model_path, self._load_model).build_discrete_lambda(input_shape, critic_alpha, 200,
                                                                                   discount, lambda_v, cont_alpha,
                                                                                   epochs)
        actor.set_history_listener(history_listener=self._actor_history_listener)
        critic.set_history_listener(history_listener=self._critic_history_listener)

        return Dyna(models=[actor, critic], algorithm=algorithm)

    def _actor_history_listener(self, history, steps):
        NNBasicModel.add_history_loss_average(history, steps, self._actor_losses)

    def _critic_history_listener(self, history, steps):
        NNBasicModel.add_history_loss_average(history, steps, self._critic_losses)

    def reward_listener(self, reward):
        self._rewards.append(reward)
        PlotHelper.plot(self._actor_losses, "Actor", self._axis[0], "r-")
        PlotHelper.plot(self._rewards, "Rewards", self._axis[2], "y-")
        PlotHelper.plot(self._critic_losses, "Critic", self._axis[1], "b-")
        PlotHelper.draw_all()
