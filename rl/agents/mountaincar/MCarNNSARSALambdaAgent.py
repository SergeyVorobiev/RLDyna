from collections import deque

import numpy as np
import tensorflow as tf
from gym import Env
from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.dyna.Dyna import Dyna
from rl.helpers.PlotHelper import PlotHelper
from rl.helpers.TerminalCheckerHelper import TerminalCheckerHelper
from rl.models.NNBasicModel import NNBasicModel
from rl.models.presets.LSARSA import LSARSA
from rl.policy.EGreedyRPolicy import EGreedyRPolicy
from rl.tasks.mountaincar.MountainCarRewardEstimator import MountainCarRewardEstimator


# Episodic semi-gradient SARSA with Eligibility Traces
class MCarNNSARSALambdaAgent(RDynaAgentBuilder):

    def __init__(self, model_path=None, load_model=False, test_mode=False):
        self._model_path = model_path
        self._load_model = load_model
        self._test_mode = test_mode
        size = 200
        self._losses = deque(maxlen=size)
        self._rewards = deque(maxlen=size)
        self._fig, self._axis = PlotHelper.build_subplots(2, 1000, 50, width=8, height=4)
        self._sarsa = None
        self._build3d_grid()

    def _build3d_grid(self):
        self._fig2, self._ax = PlotHelper.build_3d(1000, 500, width=8, height=4)
        self._grid_size = 30
        self._xs = np.linspace(-1.1, 0.52, self._grid_size)
        self._ys = np.linspace(-0.055, 0.055, self._grid_size)
        self.X, self.Y = np.meshgrid(self._xs, self._ys)
        self._states = []
        for i in range(self._grid_size):
            for j in range(self._grid_size):
                x = self.X[i][j]
                y = self.Y[i][j]
                st = [x, y]
                self._states.append(st)

    def build_agent(self, env: Env):
        alpha = 0.0001
        epochs = 1
        n_steps = 1
        lambda_v = 0.97
        discount = 1

        # It will send the bunch of samples to nn to learn like 1-5, 5-10, 10-15 instead of 1-5, 2-6, 3-7 to spead up
        # the process
        clear_memory_every_n_steps = True

        # This flag converts SARSA into Q by adding maxQ tail to the end step.
        use_max_q = False

        # You probably need to change epsilon manually, depending on the level of learning after each run
        # between 0.3 - 0.01
        e_greedy = EGreedyRPolicy(0.01, threshold=0.01, improve_step=0.00002)

        sarsa_builder = LSARSA(alpha, discount, lambda_v, self._model_path, self._load_model)

        self._sarsa = sarsa_builder.build_discrete(input_shape=(2,),
                                                   actions=env.action_space.n,
                                                   size=200,
                                                   act_common="tanh",
                                                   act_action="tanh",
                                                   l1=0.01,
                                                   l2=0.01,
                                                   dropout=0.01,
                                                   batch_normalization=False,
                                                   epochs=epochs)

        self._sarsa.set_history_listener(history_listener=self._history_listener)

        algorithm = sarsa_builder.build_algorithm(e_greedy, n_steps, clear_memory_every_n_steps, use_max_q,
                                                  TerminalCheckerHelper.truncated_std_checker)

        return Dyna(models=[self._sarsa], algorithm=algorithm, test_mode=self._test_mode,
                    reward_estimator=MountainCarRewardEstimator())

    def reward_listener(self, reward):
        self._rewards.append(reward)
        PlotHelper.plot(self._losses, "Actor", self._axis[0], "r-")
        PlotHelper.plot(self._rewards, "Reward", self._axis[1], "y-")
        self._plot3d()
        PlotHelper.draw_all()

    def _history_listener(self, history, steps):
        NNBasicModel.add_history_loss_average(history, steps, self._losses)

    def _plot3d(self):
        result = self._sarsa.predict(self._states)
        result = tf.reduce_max(result, axis=1)
        z = []
        for i in range(self._grid_size):
            zi = []
            for j in range(self._grid_size):
                index = i * self._grid_size + j
                zi.append(float(result[index]))
            z.append(zi)
        self._ax.clear()
        self._ax.plot_surface(self.X, self.Y, np.array(z), cmap='viridis')

