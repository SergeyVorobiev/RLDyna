from rl.agents.RDynaAgentBuilder import RDynaAgentBuilder
from rl.algorithms.NSARSA import NSARSA
from rl.dyna.Dyna import Dyna
from rl.environments.fl.BasicGridEnv import BasicGridEnv, StateType
from rl.models.CNNQModel import CNNQModel
from rl.planning.HashPlanning import HashPlanning
from rl.policy.EGreedyRPolicy import EGreedyRPolicy
from rl.tasks.fl.FrozenLakeCNNStatePrepare import FrozenLakeCNNStatePrepare


class CNNNSARSAAgent(RDynaAgentBuilder):

    def __init__(self, model_path=None, load_model=False):
        self._model_path = model_path
        self._load_model = load_model

    def build_agent(self, env: BasicGridEnv):
        env.set_state_type(StateType.all_map_and_around)
        n_states = env.get_x() * env.get_y()
        discount = 1
        alpha = 1
        steps = 5

        e_greedy = EGreedyRPolicy(0.4, threshold=0.02, improve_step=0.02)

        planning = HashPlanning(plan_batch_size=200, plan_step_size=200,
                                memory_size=200)

        algorithm = NSARSA(e_greedy, alpha=alpha, discount=discount, n_step=steps)

        # Batch size currently is not used, as we use only hash unique states
        models = [CNNQModel(input_shape=(env.get_y(), env.get_x(), 1), n_actions=env.action_space.n,
                            batch_size=0, epochs=20, steps_to_train=500, hash_unique_states_capacity=200,
                            model_path=self._model_path, load_model=self._load_model)]

        return Dyna(models=models, algorithm=algorithm, planning=planning,
                    state_prepare=FrozenLakeCNNStatePrepare())
