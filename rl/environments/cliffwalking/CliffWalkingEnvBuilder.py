import os
from collections import deque
from enum import Enum
from typing import Any
from gym import Env

from rl.ProjectPath import ProjectPath
from rl.agents.cliffwalking.CWTabularQAgent import CWTabularQAgent
from rl.agents.cliffwalking.CWTabularSARSAAgent import CWTabularSARSAAgent
from rl.environments.fl.BasicGridEnv import DrawInfo
from rl.environments.fl.FrozenLakeEnv import FrozenLakeEnv
from rl.dyna.Dyna import Dyna
from rl.environments.EnvBuilder import EnvBuilder

# ================================================ CONTROL PANEL =======================================================

iterations = 1000000

env_name = "CliffWalking"

# move, fall in hole, hit a wall, finish
rewards = [-1.0, -50.0, -1.0, 0.0]

color_map = [[200, 200, 255], [60, 60, 60], [0, 0, 0], [255, 200, 0], [0, 255, 0], [255, 0, 0]]

need_colorize_q_map = True

draw_info = DrawInfo.full

skip_frames = 5  # to speed up computations

colorize_q_map_frames_skip = 20  # to speed up computations

slow_render = 0  # to see what is going on more carefully

begin_from_start_if_get_in_hole = True

grid_map = [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3]], 120, 80]


# ======================================================================================================================


class CliffWalkingMethod(Enum):
    SARSA = 0,
    Q = 1,


def get_agent(method: CliffWalkingMethod, model_suffix, need_to_load):
    model_name = method.name + "_" + str(int(model_suffix))
    path = ProjectPath.join_to_table_models_path(os.path.join(env_name, model_name))
    path_nn = ProjectPath.join_to_nn_models_path(os.path.join(env_name, model_name))

    if method == CliffWalkingMethod.SARSA:
        return CWTabularSARSAAgent()
    elif method == CliffWalkingMethod.Q:
        return CWTabularQAgent()
    return None


class CliffWalkingEnvBuilder(EnvBuilder):

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
        self._save_each_episodes = 100
        self._standard = 500

    def get_iterations(self):
        return iterations

    def episode_done(self, player_prop: Any):
        self._episodes += 1

    def iteration_complete(self, state, action, reward, next_state, done, truncated, player_prop):
        self._ep_iter += 1

    def stop_render(self):
        return self._is_env_closed()

    def build_env_and_agent(self) -> (Env, Dyna):
        env = FrozenLakeEnv(rewards=rewards, grid_map=grid_map[0], cell_width=grid_map[1], cell_height=grid_map[2],
                            color_map=color_map, begin_from_start_if_get_in_hole=begin_from_start_if_get_in_hole)
        env.set_name("CliffWalking")
        self._is_env_closed = env.is_closed

        # Skip some part of screen updating to speed up the process.
        env.set_skip_frame(skip_frames)
        env.draw_map_frame_skip(colorize_q_map_frames_skip)
        env.set_slow_render(slow_render)
        env.draw_info(draw_info)
        env.draw_map(draw_map=need_colorize_q_map)

        # build_acd Dyna agent
        self._agent = get_agent(self._method, self._model_suffix, self._need_to_load).build_agent(env)
        env.draw_values_setup(q_supplier=self._agent.get_q_values)

        return env, self._agent
