from collections import deque

from gym import Env
from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.dyna.Dyna import Dyna
from rl.helpers.PlotHelper import PlotHelper
from rl.models.NNBasicModel import NNBasicModel
from rl.models.presets.LSARSA import LSARSA
from rl.policy.EGreedyRPolicy import EGreedyRPolicy


# Episodic semi-gradient SARSA with Eligibility Traces
class LunarLanderNNSARSALambdaAgent(RDynaAgentBuilder):

    def __init__(self, model_path=None, load_model=False, test_mode=False):
        self._model_path = model_path
        self._load_model = load_model
        self._test_mode = test_mode
        self._build_nn = None

        stat_size = 200
        self._action1 = deque(maxlen=stat_size)
        self._action2 = deque(maxlen=stat_size)
        self._action3 = deque(maxlen=stat_size)
        self._action4 = deque(maxlen=stat_size)
        self._losses = deque(maxlen=stat_size)
        self._rewards = deque(maxlen=1000)
        self._fig, self._axis = PlotHelper.build_subplots(6, 1000, 50)

    def build_agent(self, env: Env):
        alpha = 0.00001
        lambda_v = 0.0
        discount = 1
        n_steps = 20
        e_greedy = EGreedyRPolicy(0.2, threshold=0.01, improve_step=0.00002)

        # It will send the bunch of samples to nn to learn like 1-5, 5-10, 10-15 instead of 1-5, 2-6, 3-7 to spead up
        # the process
        clear_memory_every_n_steps = True

        sarsa_builder = LSARSA(alpha, discount, lambda_v, self._model_path, self._load_model)

        sarsa = sarsa_builder.build_discrete(input_shape=(8,),
                                             actions=env.action_space.n,
                                             size=400,
                                             act_common="tanh",
                                             act_action="relu",
                                             l1=0.0001,
                                             l2=0.0001,
                                             batch_normalization=False,
                                             epochs=2)

        algorithm = sarsa_builder.build_algorithm(e_greedy, n_steps, clear_memory_every_n_steps, False, None,
                                                  self._actions_listener)

        sarsa.set_history_listener(history_listener=self._history_listener)

        return Dyna(models=[sarsa], algorithm=algorithm, test_mode=self._test_mode)

    def reward_listener(self, reward):
        self._rewards.append(reward)
        PlotHelper.plot(self._losses, "Actor", self._axis[0], "r-")
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

    def _history_listener(self, history, steps):
        NNBasicModel.add_history_loss_average(history, steps, self._losses)
