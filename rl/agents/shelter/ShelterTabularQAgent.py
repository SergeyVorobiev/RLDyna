from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.algorithms.QAlgorithm import QAlgorithm
from rl.dyna.Dyna import Dyna
from rl.environments.fl.BasicGridEnv import BasicGridEnv, StateType
from rl.models.Table1D import Table1D
from rl.policy.EGreedyRPolicy import EGreedyRPolicy
from rl.tasks.shelter.ShelterStatePrepare import ShelterStatePrepare


class ShelterTabularQAgent(RDynaAgentBuilder):

    def build_agent(self, env: BasicGridEnv):
        env.set_state_type(StateType.blind)
        n_states = env.get_x() * env.get_y()
        discount = 1

        alpha = 1

        e_greedy = EGreedyRPolicy(0.1, threshold=0.001, improve_step=0.001)

        algorithm = QAlgorithm(e_greedy, alpha=alpha, discount=discount)

        models = [Table1D(n_states=n_states, n_actions=env.action_space.n)]

        return Dyna(models=models, algorithm=algorithm, state_prepare=ShelterStatePrepare(env.get_y()))
