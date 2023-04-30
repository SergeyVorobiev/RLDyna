from rl.algorithms.NNSARSAAlgorithm import NNSARSAAlgorithm
from rl.helpers.ModelSaveLoadHelper import ModelSaveLoadHelper
from rl.models.NNBasicModel import NNBasicModel
from rl.models.SARSAModelForNN import SARSAModelForNN
from rl.models.losses.CustomLoss import CustomLoss
from rl.models.nnbuilders.CustomNetwork import CustomNetwork
from rl.models.nns.NNSARSALambda import NNSARSALambda


# Neural network sarsa with eligibility traces for discrete action space
class LSARSA:

    def __init__(self, alpha, discount, lambda_v, model_path, load_model):
        self._model_path = model_path
        self._load_model = load_model
        self._alpha = alpha
        self._lambda = lambda_v
        self._discount = discount

    def build_algorithm(self, policy, n_steps, clear_memory_every_n_steps, use_max_q, terminal_state_checker=None,
                        actions_listener=None):
        return NNSARSAAlgorithm(policy=policy, alpha=self._alpha, discount=self._discount, memory_capacity=n_steps,
                                clear_memory_every_n_steps=clear_memory_every_n_steps,
                                next_q_max=use_max_q, terminal_state_checker=terminal_state_checker,
                                actions_listener=actions_listener)

    def build_discrete(self, input_shape, actions, size, act_common, act_action, l1, l2, batch_normalization,
                       epochs, verbose=0, dropout=0.0) -> NNBasicModel:
        def build_model_func(inputs, outputs):
            return NNSARSALambda(inputs, outputs, self._lambda, self._discount)
            # return NNSARSA(inputs, outputs)

        def build_nn():
            return CustomNetwork.discrete_separate(input_shape=input_shape,
                                                   output_n=actions,
                                                   alpha=self._alpha,
                                                   size=size,
                                                   out="linear",
                                                   custom_model_build_func=build_model_func,
                                                   act_common=act_common,
                                                   act_action=act_action,
                                                   l1=l1,
                                                   l2=l2,
                                                   dropout=dropout,
                                                   loss=CustomLoss.q_loss,
                                                   batch_normalization=batch_normalization,
                                                   run_eagerly=True)

        def custom_load_model_func(path):
            return ModelSaveLoadHelper.load_weights_h5(path, build_nn)

        return SARSAModelForNN(n_actions=actions, nn_build_function=build_nn, model_path=self._model_path,
                               load_model=self._load_model,
                               custom_load_model_func=custom_load_model_func,
                               custom_save_model_func=ModelSaveLoadHelper.save_weights_h5,
                               epochs=epochs, verbose=verbose)
