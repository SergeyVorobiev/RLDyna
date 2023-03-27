from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.algorithms.Q import Q
from rl.dyna.Dyna import Dyna
from rl.environment.BasicGridEnv import BasicGridEnv, StateType
from rl.models.CNNQModel import CNNQModel
from rl.planning.SimplePlanning import SimplePlanning
from rl.policy.EGreedyRPolicy import EGreedyRPolicy
from rl.tasks.fl.FrozenLakeCNNStatePrepare import FrozenLakeCNNStatePrepare


class CNNQAgent(RDynaAgentBuilder):

    def build_agent(self, env: BasicGridEnv):
        env.set_state_type(StateType.all_map_and_around)
        n_states = env.get_x() * env.get_y()
        discount = 1
        alpha = 1

        e_greedy = EGreedyRPolicy(0.4, threshold=0.02, improve_step=0.02)

        # memory_size - Capacity of memory, if number of lines is exceeded, then just forget oldest.
        # plan_batch_size - Number of lines that needs to be randomly obtained from the memory to train.
        # plan_step_size - Number of steps that needs to be passed to start planning process.
        planning = SimplePlanning(plan_batch_size=n_states, plan_step_size=n_states,
                                  memory_size=n_states * env.action_space.n)

        # Iterative algorithm
        algorithm = Q(e_greedy, alpha=alpha, discount=discount)

        # Batch size currently is not used, as we use only hash unique states
        models = [CNNQModel(input_shape=(env.get_y(), env.get_x(), 1), n_actions=env.action_space.n, batch_size=0,
                            epochs=10, steps_to_train=100, hash_unique_states_capacity=200)]

        return Dyna(models=models, algorithm=algorithm, planning=planning,
                    state_prepare=FrozenLakeCNNStatePrepare())
