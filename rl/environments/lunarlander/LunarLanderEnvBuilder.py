import os
from collections import deque
from typing import Any
from gym import Env
from rl.ProjectPath import ProjectPath
from rl.agents.lunarlander.LunarLanderMCPGTDActorCriticAgent import LunarLanderMCPGTDActorCriticAgent
from rl.dyna.Dyna import Dyna
from rl.environments.EnvBuilder import EnvBuilder
import gym

# ================================================ CONTROL PANEL =======================================================

iterations = 1000000

env_name = "LunarLander"

selected_agent = "MCPGTDActorCritic"  # MCPGTDActorCritic

model_name_suffix = "1"

model_name = selected_agent + "_" + model_name_suffix

path = ProjectPath.join_to_table_models_path(os.path.join(env_name, model_name))
path_nn = ProjectPath.join_to_nn_models_path(os.path.join(env_name, model_name))

load_model = True
save_model = True

agents = {
    "MCPGTDActorCritic": LunarLanderMCPGTDActorCriticAgent(path_nn, load_model)
}


# ======================================================================================================================


class LunarLanderEnvBuilder(EnvBuilder):

    def __init__(self):
        self._ep_iter = 0
        self._episodes = 0
        self._save_each_episodes = 20
        self._algorithm_memory = 0
        self._max_len = 1000
        self._rewards = deque(maxlen=self._max_len)
        self._reward = 0
        self._max_reward = -1000000

    def get_iterations(self):
        return iterations

    def episode_done(self, player_prop: Any):
        self._episodes += 1
        if self._max_reward < self._reward:
            self._max_reward = self._reward
        str_reward = str(round(self._reward, 1))
        str_max_reward = str(round(self._max_reward, 1))
        self._rewards.append(self._reward)
        aver_rew = sum(self._rewards) / self._rewards.__len__()
        str_aver_rew = str(round(aver_rew, 1))
        print(f"Episode {self._episodes} Reward: {str_reward} Max Reward: {str_max_reward} "
              f"RewardAver{self._max_len}: {str_aver_rew}")

        if self._ep_iter > self._algorithm_memory:
            print(f"Warning! You use Monte Carlo with {self._algorithm_memory} "
                  f"memory capacity but episode has {self._ep_iter}")

        if self._episodes % self._save_each_episodes == 0 and save_model:
            EnvBuilder.save_model(self._agent)

        self._ep_iter = 0
        self._reward = 0

    def iteration_complete(self, state, action, reward, next_state, done, player_prop):
        self._ep_iter += 1
        self._reward += reward

    def stop_render(self):
        pass

    def build_env_and_agent(self) -> (Env, Dyna):
        env = gym.make("LunarLander-v2")
        self._agent = agents[selected_agent].build_agent(env)
        self._algorithm_memory = self._agent.get_algorithm_memory_capacity()
        return env, self._agent
