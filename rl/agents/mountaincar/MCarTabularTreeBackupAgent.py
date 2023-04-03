from gym import Env

from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.algorithms.TreeBackup import TreeBackup
from rl.dyna.Dyna import Dyna
from rl.models.DiscreteQTable import DiscreteQTable
from rl.planning.NoPlanning import NoPlanning
from rl.policy.EGreedyRPolicy import EGreedyRPolicy
from rl.tasks.mountaincar.MountainCarDiscreteStatePrepare import MountainCarDiscreteStatePrepare
from rl.tasks.mountaincar.MountainCarRewardEstimator import MountainCarRewardEstimator


class MCTabularTreeBackupAgent(RDynaAgentBuilder):

    def __init__(self, model_path=None, load_model=False):
        self._model_path = model_path
        self._load_model = load_model

    def build_agent(self, env: Env):
        discount = 0.999
        alpha = 1 / 20
        e_greedy = EGreedyRPolicy(0.001, threshold=0.001, improve_step=0.0001)
        planning = NoPlanning()

        # Iterative algorithm
        algorithm = TreeBackup(e_greedy, alpha=alpha, discount=discount, n_step=100)
        dim_min_max = [[-1.2, 0.54],
                       [-0.06, 0.06]]
        table = DiscreteQTable(dim_min_max=dim_min_max, quant_size=10, n_actions=env.action_space.n,
                               model_path=self._model_path, load_model=self._load_model)
        models = [table]

        return Dyna(models=models, algorithm=algorithm, planning=planning,
                    state_prepare=MountainCarDiscreteStatePrepare(table.digitize_state),
                    reward_estimator=MountainCarRewardEstimator())
