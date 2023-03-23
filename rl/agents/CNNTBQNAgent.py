from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.algorithms.TreeBackup import TreeBackup
from rl.dyna.Dyna import Dyna
from rl.environment.BasicGridEnv import BasicGridEnv, StateType
from rl.models.CNNQModel import CNNQModel, MaxQ
from rl.planning.SimplePlanning import SimplePlanning
from rl.policy.EGreedyRPolicy import EGreedyRPolicy
from rl.tasks.fl.FrozenLakeCNNStatePrepare import FrozenLakeCNNStatePrepare


class CNNTBQNAgent(RDynaAgentBuilder):

    def build_agent(self, env: BasicGridEnv):

        env.set_state_type(StateType.all_map)
        n_states = env.get_x() * env.get_y()
        discount = 1
        alpha = 1
        steps = 3

        e_greedy = EGreedyRPolicy(0.4, threshold=0.02, improve_step=0.02)
        planning = SimplePlanning(plan_batch_size=n_states, plan_step_size=n_states,
                                  memory_size=n_states * env.action_space.n)
        # planning = NoPlanning()
        algorithm = TreeBackup(e_greedy, alpha=alpha, discount=discount, n_step=steps)
        models = [CNNQModel(input_shape=(env.get_y(), env.get_x(), 1), n_actions=env.action_space.n,
                            batch_size=1000, epochs=30, steps_to_train=400, max_q=MaxQ.from_support)]

        return Dyna(models=models, algorithm=algorithm, planning=planning,
                    state_prepare=FrozenLakeCNNStatePrepare())
