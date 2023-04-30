import os
from collections import deque
from enum import Enum
from typing import Any
from gym import Env
from rl.ProjectPath import ProjectPath
from rl.agents.lunarlandercont.LunarLanderMCPGCAgent import LunarLanderMCPGCAgent
from rl.agents.lunarlandercont.LunarLanderTDPGCACAgent import LunarLanderTDPGCACAgent
from rl.dyna.Dyna import Dyna
from rl.environments.EnvBuilder import EnvBuilder
import gym

# ================================================ CONTROL PANEL =======================================================

iterations = 1000000

env_name = "LunarLanderCont"


# ======================================================================================================================


class LunarLanderContMethod(Enum):
    MCPGC = 0
    TDPGCAC = 1


def get_agent(method: LunarLanderContMethod, model_suffix, need_to_load):
    model_name = method.name + "_" + str(int(model_suffix))
    path = ProjectPath.join_to_table_models_path(os.path.join(env_name, model_name))
    path_nn = ProjectPath.join_to_nn_models_path(os.path.join(env_name, model_name))
    if method == LunarLanderContMethod.MCPGC:
        return LunarLanderMCPGCAgent(path_nn, need_to_load)
    if method == LunarLanderContMethod.TDPGCAC:
        return LunarLanderTDPGCACAgent(path_nn, need_to_load)
    return None


class LunarLanderContEnvBuilder(EnvBuilder):

    def __init__(self, model_suffix, need_to_load, need_to_save, method):
        self._model_suffix = model_suffix
        self._need_to_load = need_to_load
        self._need_to_save = need_to_save
        self._method = method
        self._ep_iter = 0
        self._episodes = 0
        self._save_each_episodes = 10
        self._reward_sum = 0
        self._max_reward = -100000
        self._max_av_rewards_len = 1000
        self._rewards = deque(maxlen=self._max_av_rewards_len)
        self._agent_builder = None
        self._agent = None

    def get_iterations(self):
        return iterations

    def episode_done(self, player_prop: Any):
        self._episodes += 1
        if self._max_reward < self._reward_sum:
            self._max_reward = self._reward_sum
        reward_str = str(round(self._reward_sum, 3))
        self._agent_builder.reward_listener(self._reward_sum)
        max_reward_str = str(round(self._max_reward, 3))
        self._rewards.append(self._reward_sum)
        aver_r = sum(self._rewards) / self._rewards.__len__()
        aver_r_str = str(round(aver_r, 3))
        print(f"Episode: {self._episodes}   Reward: {reward_str}   Average{self._max_av_rewards_len}: {aver_r_str}   "
              f"Max Reward: {max_reward_str}")
        self._ep_iter = 0
        self._reward_sum = 0
        if self._episodes % self._save_each_episodes == 0 and self._need_to_save:
            EnvBuilder.save_model(self._agent)

    def iteration_complete(self, state, action, reward, next_state, done, truncated, player_prop):
        self._ep_iter += 1
        self._reward_sum += reward

    def stop_render(self):
        pass

    def build_env_and_agent(self) -> (Env, Dyna):
        env = gym.make("LunarLanderContinuous-v2")
        self._agent_builder = get_agent(self._method, self._model_suffix, self._need_to_load)
        self._agent = self._agent_builder.build_agent(env)
        return env, self._agent
