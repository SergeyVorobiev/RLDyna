import os
from collections import deque
from typing import Any

from gym import Env

from rl.ProjectPath import ProjectPath
from rl.agents.cartpole.CPTabularQAgent import CPTabularQAgent
from rl.dyna.Dyna import Dyna
from rl.environments.EnvBuilder import EnvBuilder
import gym


# ================================================ CONTROL PANEL =======================================================

iterations = 1000000

env_name = "CartPole"

selected_agent = "TabQ"

path = ProjectPath.join_to_table_models_path(os.path.join(env_name, selected_agent))

load_model = True
save_model = True

agents = {
    "TabQ": CPTabularQAgent(path, load_model)
}

# ======================================================================================================================


class CartPoleEnvBuilder(EnvBuilder):

    def __init__(self):
        self._agent_name = selected_agent
        self._ep_iter = 0
        self._episodes = 0
        self._score_average = deque(maxlen=100)
        self._save_each_episodes = 100

    def get_iterations(self):
        return iterations

    def episode_done(self, player_prop: Any):
        self._episodes += 1
        self._score_average.append(self._ep_iter)
        average = sum(self._score_average) / self._score_average.__len__()
        print("Episode " + str(self._episodes) + ": " + str(self._ep_iter) + " Average100: " + str(average))
        self._ep_iter = 0

        if self._episodes % self._save_each_episodes == 0:
            if save_model and self._agent is not None:
                if self._agent.get_models()[0].save():
                    print("Model is saved")

    def iteration_complete(self, player_prop: Any):
        self._ep_iter += 1

    def stop_render(self):
        pass

    def build_env_and_agent(self) -> (Env, Dyna):
        env = gym.make("CartPole-v1")
        self._agent = agents[self._agent_name].build_agent(env)
        return env, self._agent
