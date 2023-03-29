from gym import Env

from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.algorithms.Q import Q
from rl.dyna.Dyna import Dyna
from rl.models.DiscreteQTable import DiscreteQTable
from rl.planning.NoPlanning import NoPlanning
from rl.policy.EGreedyRPolicy import EGreedyRPolicy
from rl.tasks.cartpole.CartPoleDiscreteStatePrepare import CartPoleDiscreteStatePrepare
from rl.tasks.cartpole.CartPoleRewardEstimator import CartPoleRewardEstimator


class CPTabularQAgent(RDynaAgentBuilder):

    def __init__(self, model_path=None, load_model=False):
        self._model_path = model_path
        self._load_model = load_model

    def build_agent(self, env: Env):

        # To avoid infinite reward growing
        discount = 0.99

        # For constant Reward alpha less than 1 only slow you down because average of [a, a, a, a, a] will always be a.
        # But here we use discrete table meaning that potentially different states will be stored in the same table cell
        # and potentially different values should be averaged.
        alpha = 1 / 10
        e_greedy = EGreedyRPolicy(0.2, threshold=0.02, improve_step=0.00001)
        planning = NoPlanning()

        # Iterative algorithm
        algorithm = Q(e_greedy, alpha=alpha, discount=discount)

        dim_min_max = [[-1.5, 1.5],
                       [-3, 3],
                       [-0.5, 0.5],
                       [-3, 3]]
        table = DiscreteQTable(dim_min_max=dim_min_max, quant_size=40, n_actions=env.action_space.n,
                               model_path=self._model_path, load_model=self._load_model)

        models = [table]

        return Dyna(models=models, algorithm=algorithm, planning=planning,
                    state_prepare=CartPoleDiscreteStatePrepare(table.digitize_state),
                    reward_estimator=CartPoleRewardEstimator(), allow_clear_memory=False)
