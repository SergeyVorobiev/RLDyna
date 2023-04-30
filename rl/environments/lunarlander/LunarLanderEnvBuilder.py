import os
from collections import deque
from enum import Enum
from typing import Any
from gym import Env
from rl.ProjectPath import ProjectPath
from rl.agents.lunarlander.LunarLanderTDPGDACAgent import LunarLanderTDPGDACAgent
from rl.agents.lunarlander.LunarLanderMCPGDLAgent import LunarLanderMCPGDLAgent
from rl.agents.lunarlander.LunarLanderNNSARSALambdaAgent import LunarLanderNNSARSALambdaAgent
from rl.dyna.Dyna import Dyna
from rl.environments.EnvBuilder import EnvBuilder
import gym

# ================================================ CONTROL PANEL =======================================================

iterations = 1000000

env_name = "LunarLander"

# ======================================================================================================================


class LunarLanderMethod(Enum):
    MCPGDL = 1
    TDPGDAC = 2
    NNSARSALambda = 3


def get_agent(method: LunarLanderMethod, model_suffix, need_to_load):
    model_name = method.name + "_" + str(int(model_suffix))
    path = ProjectPath.join_to_table_models_path(os.path.join(env_name, model_name))
    path_nn = ProjectPath.join_to_nn_models_path(os.path.join(env_name, model_name))

    if method == LunarLanderMethod.TDPGDAC:
        return LunarLanderTDPGDACAgent(path_nn, need_to_load)
    elif method == LunarLanderMethod.MCPGDL:
        return LunarLanderMCPGDLAgent(path_nn, need_to_load)
    elif method == LunarLanderMethod.NNSARSALambda:
        return LunarLanderNNSARSALambdaAgent(path_nn, need_to_load)
    return None


class LunarLanderEnvBuilder(EnvBuilder):

    def __init__(self, model_suffix, need_to_load, need_to_save, method):
        self._model_suffix = model_suffix
        self._need_to_load = need_to_load
        self._need_to_save = need_to_save
        self._method = method
        self._ep_iter = 0
        self._episodes = 0
        self._save_each_episodes = 10
        self._algorithm_memory = 0
        self._max_len = 1000
        self._rewards = deque(maxlen=self._max_len)
        self._reward = 0
        self._max_reward = -1000000
        self._agent_builder = None

    def get_iterations(self):
        return iterations

    def episode_done(self, player_prop: Any):
        self._episodes += 1
        if self._max_reward < self._reward:
            self._max_reward = self._reward
        str_reward = str(round(self._reward, 1))
        str_max_reward = str(round(self._max_reward, 1))
        self._rewards.append(self._reward)
        self._agent_builder.reward_listener(self._reward)
        aver_rew = sum(self._rewards) / self._rewards.__len__()
        str_aver_rew = str(round(aver_rew, 1))
        print(f"Episode {self._episodes} Reward: {str_reward} Max Reward: {str_max_reward} "
              f"RewardAver{self._max_len}: {str_aver_rew}")

        if self._episodes % self._save_each_episodes == 0 and self._need_to_save:
            EnvBuilder.save_model(self._agent)

        self._ep_iter = 0
        self._reward = 0

    def iteration_complete(self, state, action, reward, next_state, done, truncated, player_prop):
        self._ep_iter += 1
        self._reward += reward

    def stop_render(self):
        pass

    def build_env_and_agent(self) -> (Env, Dyna):
        env = gym.make("LunarLander-v2")
        self._agent_builder = get_agent(self._method, self._model_suffix, self._need_to_load)
        self._agent = self._agent_builder.build_agent(env)
        self._algorithm_memory = self._agent.get_algorithm_memory_capacity()
        return env, self._agent
