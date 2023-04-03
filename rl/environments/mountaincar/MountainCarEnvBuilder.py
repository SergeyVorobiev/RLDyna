import os
from typing import Any
from gym import Env
from rl.ProjectPath import ProjectPath
from rl.agents.mountaincar.MCarTabularTreeBackupAgent import MCTabularTreeBackupAgent
from rl.dyna.Dyna import Dyna
from rl.environments.EnvBuilder import EnvBuilder
import gym


# ================================================ CONTROL PANEL =======================================================

iterations = 1000000

env_name = "MountainCar"

selected_agent = "TabTBQN"  # TabTBQN

model_name_suffix = "1"

model_name = selected_agent + "_" + model_name_suffix

path = ProjectPath.join_to_table_models_path(os.path.join(env_name, model_name))
path_nn = ProjectPath.join_to_nn_models_path(os.path.join(env_name, model_name))

load_model = True
save_model = True

agents = {
    "TabTBQN": MCTabularTreeBackupAgent(path, load_model),
}

# ======================================================================================================================


class MountainCarEnvBuilder(EnvBuilder):

    def __init__(self):
        self._agent_name = selected_agent
        self._iter = 0
        self._episodes = 0
        self._save_each_episodes = 10
        self._min_iter = 200
        self._reward = None
        self._agent = None

    def get_iterations(self):
        return iterations

    def episode_done(self, player_prop: Any):
        self._episodes += 1
        if self._min_iter > self._iter:
            self._min_iter = self._iter
        r = str(round(self._reward, 3))
        print(f"Episode {self._episodes}: {self._iter}    Record: {self._min_iter}   Best Result: {r}")
        self._iter = 0

        if self._episodes % self._save_each_episodes == 0 and save_model:
            EnvBuilder.save_model(self._agent)

    def iteration_complete(self, state, action, reward, next_state, done, player_prop):
        self._iter += 1

        if done:
            reward = next_state[0]
            if self._reward is None or self._reward < reward:
                self._reward = reward

    def stop_render(self):
        pass

    def build_env_and_agent(self) -> (Env, Dyna):
        env = gym.make("MountainCar-v0")
        self._agent = agents[self._agent_name].build_agent(env)
        return env, self._agent
