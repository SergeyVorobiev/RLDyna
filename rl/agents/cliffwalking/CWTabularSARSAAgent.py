from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.algorithms.NSARSAAlgorithm import NSARSAAlgorithm
from rl.dyna.Dyna import Dyna
from rl.environments.fl.BasicGridEnv import BasicGridEnv, StateType
from rl.models.Table1D import Table1D
from rl.planning.NoPlanning import NoPlanning
from rl.policy.EGreedyRPolicy import EGreedyRPolicy
from rl.tasks.shelter.ShelterStatePrepare import ShelterStatePrepare


class CWTabularSARSAAgent(RDynaAgentBuilder):

    def build_agent(self, env: BasicGridEnv):
        env.set_state_type(StateType.blind)
        n_states = env.get_x() * env.get_y()
        discount = 0.99

        alpha = 1 / 20

        e_greedy = EGreedyRPolicy(0.2, threshold=0.05, improve_step=0.0001)
        planning = NoPlanning()
        algorithm = NSARSAAlgorithm(e_greedy, alpha=alpha, discount=discount, n_step=4)

        models = [Table1D(n_states=n_states, n_actions=env.action_space.n)]

        return Dyna(models=models, algorithm=algorithm, planning=planning, state_prepare=ShelterStatePrepare(env.get_y()))
