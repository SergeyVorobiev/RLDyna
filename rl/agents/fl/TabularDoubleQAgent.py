from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.algorithms.DoubleQAlgorithm import DoubleQAlgorithm
from rl.dyna.Dyna import Dyna
from rl.environments.fl.BasicGridEnv import BasicGridEnv, StateType
from rl.models.Table1D import Table1D
from rl.planning.SimplePlanning import SimplePlanning
from rl.policy.EGreedyRPolicy import EGreedyRPolicy
from rl.tasks.fl.FrozenLakeStatePrepare import FrozenLakeStatePrepare


class TabularDoubleQAgent(RDynaAgentBuilder):

    def build_agent(self, env: BasicGridEnv):
        env.set_state_type(StateType.blind)
        n_states = env.get_x() * env.get_y()
        discount = 1

        # So as we do not have the probability distribution of rewards (i.e. our rewards is always const), we do not
        # need alpha at all because the average will always be the same
        alpha = 1

        # Policy, 0.02 means that in 2% it will choose the random action to explore
        e_greedy = EGreedyRPolicy(0.02, threshold=0.001, improve_step=0.001)

        # memory_size - Capacity of memory, if number of lines is exceeded, then just forget oldest.
        # plan_batch_size - Number of lines that needs to be randomly obtained from the memory to train.
        # plan_step_size - Number of steps that needs to be passed to start planning process.
        planning = SimplePlanning(plan_batch_size=n_states, plan_step_size=100,
                                  memory_size=n_states * env.action_space.n)
        # planning = NoPlanning()

        # Iterative algorithm
        algorithm = DoubleQAlgorithm(e_greedy, alpha=alpha, discount=discount)

        # Model keeps the previously learned information and get the data back when needed.
        models = [Table1D(n_states=n_states, n_actions=env.action_space.n),
                  Table1D(n_states=n_states, n_actions=env.action_space.n)]

        return Dyna(models=models, algorithm=algorithm, planning=planning,
                    state_prepare=FrozenLakeStatePrepare(env.get_y()))
