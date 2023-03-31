import os
from time import sleep
from typing import Any

from gym import Env

from rl.ProjectPath import ProjectPath
from rl.agents.mountaincar.MCTabularTreeBackupAgent import MCTabularTreeBackupAgent
from rl.dyna.Dyna import Dyna
from rl.environments.EnvBuilder import EnvBuilder
import gym


# ================================================ CONTROL PANEL =======================================================

iterations = 1000000

env_name = "MountainCar"

selected_agent = "TabTBQN"  # TabTBQN

path = ProjectPath.join_to_table_models_path(os.path.join(env_name, selected_agent))

load_model = True
save_model = True

agents = {
    "TabTBQN": MCTabularTreeBackupAgent(path, load_model)
}

# ======================================================================================================================


class MountainCarEnvBuilder(EnvBuilder):

    def __init__(self):
        self._agent_name = selected_agent
        self._iter = 0
        self._episodes = 0
        self._save_each_episodes = 10
        self._min_iter = 200

    def get_iterations(self):
        return iterations

    def episode_done(self, player_prop: Any):
        self._episodes += 1
        if self._min_iter > self._iter:
            self._min_iter = self._iter
        print(f"Episode {self._episodes}: {self._iter}    Record: {self._min_iter}")
        self._iter = 0

        if self._episodes % self._save_each_episodes == 0:
            if save_model and self._agent is not None:
                if self._agent.get_models()[0].save():
                    print("Model is saved")

    def iteration_complete(self, player_prop: Any):
        self._iter += 1

    def stop_render(self):
        pass

    def build_env_and_agent(self) -> (Env, Dyna):
        env = gym.make("MountainCar-v0")
        self._agent = agents[self._agent_name].build_agent(env)
        return env, self._agent

    def lookup_listener(self, state, action, reward, next_state, done, player_prop):
        pass
