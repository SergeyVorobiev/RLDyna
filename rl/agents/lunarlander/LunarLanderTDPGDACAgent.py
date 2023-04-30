from collections import deque
from gym import Env
from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.algorithms.MCPGDACAlgorithm import MCPGDACAlgorithm
from rl.dyna.Dyna import Dyna
from rl.helpers.PlotHelper import PlotHelper
from rl.models.NNBasicModel import NNBasicModel
from rl.models.losses.CustomLoss import CustomLoss
from rl.models.presets.MCPGDAC import MCPGDAC


# Monte Carlo Policy Gradient Discrete + Baseline Actor Critic
class LunarLanderTDPGDACAgent(RDynaAgentBuilder):

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
        self._fig, self._axis = PlotHelper.build_subplots(7, 1000, 50)

    def build_agent(self, env: Env):
        actions = env.action_space.n
        actor_alpha = 0.00001
        critic_alpha = 0.00001
        discount = 1
        actor_epochs = 5
        critic_epochs = 5
        verbose = 0

        n_steps = 20
        clear_memory_every_n_steps = True

        def get_tail(models, n_state):
            return float(models[1].predict([n_state]))

        # If memory capacity is less than the size of the episode it will be considered as TD(N)
        algorithm = MCPGDACAlgorithm(discount=discount, memory_capacity=n_steps, tail_method=get_tail,
                                     clear_memory_every_n_steps=clear_memory_every_n_steps,
                                     actions_listener=self._actions_listener)

        # By default it loads TDU for critic & PGD for actor
        actor, critic = MCPGDAC().build_acd(input_shape=(8,),
                                            n_actions=actions,
                                            actor_alpha=actor_alpha,
                                            critic_alpha=critic_alpha,
                                            actor_layer_size=500,
                                            critic_layer_size=500,
                                            actor_activation=["tanh", "tanh"],
                                            critic_activation=["relu", "relu"],
                                            actor_loss=CustomLoss.mcpgd,
                                            critic_loss=CustomLoss.mc_loss,
                                            actor_epochs=actor_epochs,
                                            critic_epochs=critic_epochs,
                                            run_eagerly=False,
                                            verbose=verbose,
                                            model_path=self._model_path,
                                            load_model=self._load_model)

        actor.set_history_listener(history_listener=self._actor_history_listener)
        critic.set_history_listener(history_listener=self._critic_history_listener)

        return Dyna(models=[actor, critic], algorithm=algorithm)

    def reward_listener(self, reward):
        self._rewards.append(reward)
        PlotHelper.plot(self._actor_losses, "Actor", self._axis[0], "r-")
        PlotHelper.plot(self._critic_losses, "Critic", self._axis[1], "b-")
        PlotHelper.plot(self._rewards, "Reward", self._axis[2], "y-")
        PlotHelper.plot(self._action1, "Action1", self._axis[3], "b-")
        PlotHelper.plot(self._action2, "Action2", self._axis[4], "b-")
        PlotHelper.plot(self._action3, "Action3", self._axis[5], "b-")
        PlotHelper.plot(self._action4, "Action4", self._axis[6], "b-")
        PlotHelper.draw_all()

    def _actor_history_listener(self, history, steps):
        NNBasicModel.add_history_loss_average(history, steps, self._actor_losses)

    def _critic_history_listener(self, history, steps):
        NNBasicModel.add_history_loss_average(history, steps, self._critic_losses)

    def _actions_listener(self, actions):
        self._action1.append(actions[0])
        self._action2.append(actions[1])
        self._action3.append(actions[2])
        self._action4.append(actions[3])
