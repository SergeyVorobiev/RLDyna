import os
from collections import deque
from typing import Any

from gym import Env

from rl.ProjectPath import ProjectPath
from rl.agents.cartpole.CPTabularQAgent import CPTabularQAgent
from rl.agents.cartpole.CPTabularTreeBackupAgent import CPTabularTreeBackupAgent
from rl.dyna.Dyna import Dyna
from rl.environments.EnvBuilder import EnvBuilder
import gym


# ================================================ CONTROL PANEL =======================================================

iterations = 1000000

env_name = "CartPole"

selected_agent = "TabTBQN"  # TabQ, TabTBQN

path = ProjectPath.join_to_table_models_path(os.path.join(env_name, selected_agent))

load_model = True
save_model = True

agents = {
    "TabQ": CPTabularQAgent(path, load_model),
    "TabTBQN": CPTabularTreeBackupAgent(path, load_model)
}

# ======================================================================================================================


class CartPoleEnvBuilder(EnvBuilder):

    def __init__(self):
        self._agent_name = selected_agent
        self._ep_iter = 0
        self._episodes = 0
        self._average_count = 1000
        self._scores = deque(maxlen=self._average_count)
        self._score_averages = deque(maxlen=self._average_count)
        self._gaps = deque(maxlen=self._average_count)
        self._save_each_episodes = 100
        self._standard = 500

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

    def lookup_listener(self, state, action, reward, next_state, done, player_prop):
        pass

