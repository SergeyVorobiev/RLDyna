import os

from gym import Env
from keras.initializers import initializers
from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.algorithms.NNSARSALambdaAlgorithm import NNSARSALambdaAlgorithm
from rl.dyna.Dyna import Dyna
from rl.models.SARSALambdaModel import SARSALambdaModel
from rl.models.losses.CustomLoss import CustomLoss
from rl.models.nnbuilders.CustomNetwork import CustomNetwork
from rl.models.nns.NNSARSALambda import NNSARSALambda
from rl.policy.EGreedyRPolicy import EGreedyRPolicy


class MCarNNSARSALambdaAgent(RDynaAgentBuilder):

    def __init__(self, model_path=None, load_model=False, test_mode=False):
        self._model_path = model_path
        self._load_model = load_model
        self._test_mode = test_mode
        self._build_nn = None
        self._lambda = 0.95
        self._discount = 1
        self._alpha = 0.0001

    def build_model_func(self, inputs, outputs):
        return NNSARSALambda(inputs, outputs, self._lambda)

    def build_agent(self, env: Env):
        actions = env.action_space.n
        alpha = 0.0001
        e_greedy = EGreedyRPolicy(0.2, threshold=0.01, improve_step=0.0002)

        self._build_nn = lambda: CustomNetwork.build_linear(input_shape=(2,), output_n=actions, alpha=alpha, size=200,
                                                            out="linear", custom_model_build_func=self.build_model_func,
                                                            act="relu", kernel_init=initializers.Constant(0),
                                                            loss=CustomLoss.one_step_sarsa_lambda)

        algorithm = NNSARSALambdaAlgorithm(policy=e_greedy, alpha=alpha, discount=self._discount)

        models = [SARSALambdaModel(n_actions=actions, nn_build_function=self._build_nn, model_path=self._model_path,
                                   load_model=self._load_model,
                                   custom_load_model_func=self._custom_load_model_func,
                                   custom_save_model_func=MCarNNSARSALambdaAgent._custom_save_model_func,
                                   )]
        return Dyna(models=models, algorithm=algorithm, test_mode=self._test_mode)

    def _custom_load_model_func(self, path):
        try:
            if os.path.exists(path + ".h5"):
                model = self._build_nn()
                model.load_weights(path + ".h5")
            else:
                return None
            return model
        except IOError as e:
            return None

    @staticmethod
    def _custom_save_model_func(model, path):
        model.save_weights(path + ".h5")
