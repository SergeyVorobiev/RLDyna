from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.algorithms.NSARSAAlgorithm import NSARSAAlgorithm
from rl.dyna.Dyna import Dyna
from rl.environments.fl.BasicGridEnv import BasicGridEnv, StateType
from rl.models.Table1D import Table1D
from rl.planning.SimplePlanning import SimplePlanning
from rl.policy.EGreedyRPolicy import EGreedyRPolicy
from rl.tasks.fl.FrozenLakeStatePrepare import FrozenLakeStatePrepare


class TabularNSARSAAgent(RDynaAgentBuilder):

    def build_agent(self, env: BasicGridEnv):

        env.set_state_type(StateType.blind)
        n_states = env.get_x() * env.get_y()

        discount = 1
        alpha = 1
        steps = 10

        e_greedy = EGreedyRPolicy(0.1, threshold=0.001, improve_step=0.001)
        planning = SimplePlanning(plan_batch_size=n_states, plan_step_size=n_states,
                                  memory_size=n_states * env.action_space.n * steps)
        algorithm = NSARSAAlgorithm(e_greedy, alpha=alpha, discount=discount, n_step=steps)
        models = [Table1D(n_states=n_states, n_actions=env.action_space.n)]

        return Dyna(models=models, algorithm=algorithm, planning=planning,
                    state_prepare=FrozenLakeStatePrepare(env.get_y()))
