from collections import deque
from gym import Env
from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.dyna.Dyna import Dyna
from rl.helpers.PlotHelper import PlotHelper
from rl.models.NNBasicModel import NNBasicModel
from rl.policy.EGreedyRPolicy import EGreedyRPolicy
from rl.models.presets.LSARSA import LSARSA


# Episodic semi-gradient SARSA with Eligibility Traces
class CartPoleNNSARSALambdaAgent(RDynaAgentBuilder):

    def __init__(self, model_path=None, load_model=False, test_mode=False):
        self._model_path = model_path
        self._load_model = load_model
        self._test_mode = test_mode
        self._build_nn = None
        self._losses = deque(maxlen=200)
        self._rewards = deque(maxlen=200)
        self._fig, self._axis = PlotHelper.build_subplots(2, 1000, 200, width=8, height=4)

    def build_agent(self, env: Env):
        n_steps = 1
        alpha = 0.001
        discount = 1
        lambda_v = 0.97
        epochs = 1

        # It will send the bunch of samples to nn to learn like 1-5, 5-10, 10-15 instead of 1-5, 2-6, 3-7 to spead up
        # the process
        clear_memory_every_n_steps = False

        # This flag converts SARSA into Q by adding maxQ tail to the end step instead of piQ.
        use_max_q = False

        e_greedy = EGreedyRPolicy(0.2, threshold=0.01, improve_step=0.0001)

        sarsa_builder = LSARSA(alpha, discount, lambda_v, self._model_path, self._load_model)

        sarsa = sarsa_builder.build_discrete(input_shape=(4,),
                                             actions=env.action_space.n,
                                             size=200,
                                             act_common="tanh",
                                             act_action="relu",
                                             l1=0.01,
                                             l2=0.01,
                                             dropout=0.01,
                                             batch_normalization=False,
                                             epochs=epochs,
                                             verbose=0)
        sarsa.set_history_listener(history_listener=self._history_listener)

        algorithm = sarsa_builder.build_algorithm(e_greedy, n_steps, clear_memory_every_n_steps, use_max_q)

        return Dyna(models=[sarsa], algorithm=algorithm, test_mode=self._test_mode)

    # Done by timeout does not mean that we have achieved the terminal state, that means we need to add a tail if not
    @staticmethod
    def _terminal_state_checker(truncated, props):
        # We suppose that if props is not empty then it contains 'TimeLimit.truncated' in this case the terminal state
        # has not been achieved
        return props.__len__() == 0

    def _history_listener(self, history, steps):
        NNBasicModel.add_history_loss_average(history, steps, self._losses)

    def reward_listener(self, reward):
        self._rewards.append(reward)
        PlotHelper.plot(self._rewards, "Rewards", self._axis[1], "y-")
        PlotHelper.plot(self._losses, "Actor", self._axis[0], "r-")
        PlotHelper.draw_all()
