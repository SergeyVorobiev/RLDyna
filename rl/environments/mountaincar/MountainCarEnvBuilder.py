import os
from enum import Enum
from typing import Any
from gym import Env
from rl.ProjectPath import ProjectPath
from rl.agents.mountaincar.MCarNNSARSALambdaAgent import MCarNNSARSALambdaAgent
from rl.agents.mountaincar.MCarTabularTreeBackupAgent import MCarTabularTreeBackupAgent
from rl.dyna.Dyna import Dyna
from rl.environments.EnvBuilder import EnvBuilder
import gym


# ================================================ CONTROL PANEL =======================================================

iterations = 1000000

env_name = "MountainCar"

test_mode = False

save_model_after_episodes = 10

# ======================================================================================================================


class MountainCarMethod(Enum):
    TabTBQN = 0
    NNSARSALambda = 1


def get_agent(method: MountainCarMethod, model_suffix, need_to_load):
    model_name = method.name + "_" + str(int(model_suffix))
    path = ProjectPath.join_to_table_models_path(os.path.join(env_name, model_name))
    path_nn = ProjectPath.join_to_nn_models_path(os.path.join(env_name, model_name))

    if method == MountainCarMethod.TabTBQN:
        return MCarTabularTreeBackupAgent(path, need_to_load)
    elif method == MountainCarMethod.NNSARSALambda:
        return MCarNNSARSALambdaAgent(path_nn, need_to_load, test_mode)
    return None


class MountainCarEnvBuilder(EnvBuilder):

    def __init__(self, model_suffix, need_to_load, need_to_save, method):
        self._model_suffix = model_suffix
        self._need_to_load = need_to_load
        self._need_to_save = need_to_save
        self._method = method
        self._iter = 0
        self._episodes = 0
        self._save_after_episodes = save_model_after_episodes
        self._min_iter = 200
        self._agent_builder = None
        self._reward = None
        self._max_reward = None
        self._agent = None

    def get_iterations(self):
        return iterations

    def episode_done(self, player_prop: Any):
        self._episodes += 1
        if self._min_iter > self._iter:
            self._min_iter = self._iter
        r = str(round(self._reward, 3))
        mr = str(round(self._max_reward, 3))
        self._agent_builder.reward_listener(self._reward)
        print(f"Episode {self._episodes}: {self._iter}  Reward: {r}   Record: {self._min_iter}   Best Reward: {mr}")
        self._iter = 0

        if self._episodes % self._save_after_episodes == 0 and self._need_to_save:
            EnvBuilder.save_model(self._agent)

    def iteration_complete(self, state, action, reward, next_state, done, truncated, player_prop):
        self._iter += 1

        if done:
            self._reward = next_state[0]
            if self._max_reward is None or self._max_reward < self._reward:
                self._max_reward = self._reward

    def stop_render(self):
        pass

    def build_env_and_agent(self) -> (Env, Dyna):
        env = gym.make("MountainCar-v0")
        self._agent_builder = get_agent(self._method, self._model_suffix, self._need_to_load)
        self._agent = self._agent_builder.build_agent(env)
        return env, self._agent
