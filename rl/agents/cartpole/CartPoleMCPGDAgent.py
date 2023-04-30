from collections import deque
from gym import Env

from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.algorithms.MCACAlgorithm import UseCritic
from rl.algorithms.MCPGDACAlgorithm import MCPGDACAlgorithm
from rl.dyna.Dyna import Dyna
from rl.helpers.PlotHelper import PlotHelper
from rl.models.NNBasicModel import NNBasicModel
from rl.models.losses.CustomLoss import CustomLoss
from rl.models.nns.NNTD import NNTD
from rl.models.presets.MCPGDAC import MCPGDAC


# Monte Carlo Policy Gradient discrete
class CartPoleMCPGDAgent(RDynaAgentBuilder):

    def __init__(self, model_path=None, load_model=False):
        self._model_path = model_path
        self._load_model = load_model
        stat_size = 200
        self._actor_losses = deque(maxlen=stat_size)
        self._critic_losses = deque(maxlen=stat_size)
        self._rewards = deque(maxlen=stat_size)
        self._fig, self._axis = PlotHelper.build_subplots(2, 1000, 200, 8, 5)

    def build_agent(self, env: Env):
        actions = env.action_space.n
        actor_alpha = 0.0001
        critic_alpha = 0.0001
        discount = 1
        epochs = 10

        # Useless for MC because memory will be cleared automatically after every episode
        # Speeds up TD(N), will send batch to learn in order 1-5, 5-10 instead of 1-5, 2-6, 3-7...
        clear_memory_every_n_steps = False

        # 500 will mean full episode MC, critic is disabled.
        memory_capacity = 500

        # It is useful if you have memory_capacity less than the episode steps in this case MC will automatically be
        # considered as TD(N) but then you need to add Unext, this method describes how to calculate the value of
        # Unext
        def get_tail(models, n_state):
            return float(models[1].predict([n_state]))

        algorithm = MCPGDACAlgorithm(discount=discount, memory_capacity=memory_capacity, tail_method=get_tail,
                                     clear_memory_every_n_steps=clear_memory_every_n_steps)

        # Disable critic if you use full episode, Enabled by default.
        algorithm.use_critic(UseCritic.No)

        actor, critic = MCPGDAC().build_acd(input_shape=(4,),
                                            n_actions=actions,
                                            actor_alpha=actor_alpha,
                                            critic_alpha=critic_alpha,
                                            critic_custom_model=NNTD,
                                            actor_loss=CustomLoss.mcpgd,
                                            critic_loss=CustomLoss.mc_loss_custom,
                                            actor_layer_size=200,
                                            critic_layer_size=200,
                                            actor_batch_normalization=False,
                                            critic_batch_normalization=False,
                                            actor_activation=["tanh", "relu"],
                                            critic_activation=["relu", "relu"],
                                            actor_epochs=epochs,
                                            critic_epochs=epochs,
                                            save_load_only_weights=True,
                                            model_path=self._model_path,
                                            load_model=self._load_model)

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
        PlotHelper.plot(self._rewards, "Rewards", self._axis[1], "y-")
        # PlotHelper.plot(self._critic_losses, "Critic", self._axis[2], "b-")
        PlotHelper.draw_all()
