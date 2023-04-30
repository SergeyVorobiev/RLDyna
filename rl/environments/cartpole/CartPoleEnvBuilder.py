import os
from collections import deque
from enum import Enum
from typing import Any
from gym import Env
from rl.ProjectPath import ProjectPath
from rl.agents.cartpole.CPTabularQAgent import CPTabularQAgent
from rl.agents.cartpole.CPTabularTreeBackupAgent import CPTabularTreeBackupAgent
from rl.agents.cartpole.CartPoleMCPGDLAgent import CartPoleMCPGDLAgent
from rl.agents.cartpole.CartPoleMCPGDAgent import CartPoleMCPGDAgent
from rl.agents.cartpole.CartPoleTDPGDLACAgent import CartPoleTDPGDLACAgent
from rl.agents.cartpole.CartPoleNNSARSALambdaAgent import CartPoleNNSARSALambdaAgent
from rl.dyna.Dyna import Dyna
from rl.environments.EnvBuilder import EnvBuilder
import gym

# ================================================ CONTROL PANEL =======================================================

iterations = 1000000

env_name = "CartPole"


# ======================================================================================================================


class CartPoleMethod(Enum):
    TabQ = 0
    TabTBQN = 1
    MCPGDL = 2
    MCPGD = 3
    TDPGDLAC = 4
    NNSARSALambda = 5


def get_agent(method: CartPoleMethod, model_suffix, need_to_load):
    model_name = method.name + "_" + str(int(model_suffix))
    path = ProjectPath.join_to_table_models_path(os.path.join(env_name, model_name))
    path_nn = ProjectPath.join_to_nn_models_path(os.path.join(env_name, model_name))

    if method == CartPoleMethod.TabQ:
        return CPTabularQAgent(path, need_to_load)
    elif method == CartPoleMethod.TabTBQN:
        return CPTabularTreeBackupAgent(path, need_to_load)
    elif method == CartPoleMethod.MCPGDL:
        return CartPoleMCPGDLAgent(path_nn, need_to_load)
    elif method == CartPoleMethod.MCPGD:
        return CartPoleMCPGDAgent(path_nn, need_to_load)
    elif method == CartPoleMethod.TDPGDLAC:
        return CartPoleTDPGDLACAgent(path_nn, need_to_load)
    elif method == CartPoleMethod.NNSARSALambda:
        return CartPoleNNSARSALambdaAgent(path_nn, need_to_load)
    return None


class CartPoleEnvBuilder(EnvBuilder):

    def __init__(self, model_suffix, need_to_load, need_to_save, method):
        self._model_suffix = model_suffix
        self._need_to_load = need_to_load
        self._need_to_save = need_to_save
        self._method = method
        self._ep_iter = 0
        self._episodes = 0
        self._average_count = 1000
        self._scores = deque(maxlen=self._average_count)
        self._score_averages = deque(maxlen=self._average_count)
        self._gaps = deque(maxlen=self._average_count)
        self._save_each_episodes = 20
        self._standard = 500
        self._agent_builder = None
        self._agent = None

    def get_iterations(self):
        return iterations

    def _get_gap(self, new_value):
        old_value = new_value
        if self._score_averages.__len__() > 0:
            old_value = self._score_averages[-1]
        result = new_value - old_value
        if result == 0:
            result = 1
        return result

    def episode_done(self, player_prop: Any):
        self._episodes += 1
        self._scores.append(self._ep_iter)
        self._agent_builder.reward_listener(self._ep_iter)
        average = sum(self._scores) / self._scores.__len__()
        gap = self._get_gap(average)
        self._gaps.append(gap)
        gap_average = sum(self._gaps) / self._gaps.__len__()
        if gap_average == 0:
            gap_average = 1
        self._score_averages.append(average)
        left_scores = self._standard - average
        epochs_count = int(left_scores / gap_average)
        if epochs_count < 0:
            epochs_count = "infinity"
        print("Episode " + str(self._episodes) + ": " + str(self._ep_iter) +
              f" Average{self._average_count}: {round(average, 2)}" +
              f"    Grow rate: {round(gap_average, 2)}   Epochs left: {epochs_count}")
        self._ep_iter = 0

        if self._episodes % self._save_each_episodes == 0 and self._need_to_save:
            EnvBuilder.save_model(self._agent)

    def iteration_complete(self, state, action, reward, next_state, done, truncated, player_prop):
        self._ep_iter += 1

    def stop_render(self):
        pass

    def build_env_and_agent(self) -> (Env, Dyna):
        env = gym.make("CartPole-v1")
        self._agent_builder = get_agent(self._method, self._model_suffix, self._need_to_load)
        self._agent = self._agent_builder.build_agent(env)
        return env, self._agent
