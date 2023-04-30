import os
from enum import Enum

from gym import Env

from rl.ProjectPath import ProjectPath
from rl.agents.fl.CNNDQAgent import CNNDQAgent
from rl.agents.fl.CNNESARSAAgent import CNNESARSAAgent
from rl.agents.fl.CNNNSARSAAgent import CNNNSARSAAgent
from rl.agents.fl.CNNQAgent import CNNQAgent
from rl.agents.fl.CNNSARSAAgent import CNNSARSAAgent
from rl.agents.fl.CNNTBQNAgent import CNNTBQNAgent
from rl.agents.fl.TabularDoubleQAgent import TabularDoubleQAgent
from rl.agents.fl.TabularESARSAAgent import TabularESARSAAgent
from rl.agents.fl.TabularNSARSAAgent import TabularNSARSAAgent
from rl.agents.fl.TabularQAgent import TabularQAgent
from rl.agents.fl.TabularSARSAAgent import TabularSARSAAgent
from rl.agents.fl.TabularTreeBackupAgent import TabularTreeBackupAgent
from rl.dyna.Dyna import Dyna
from rl.environments.EnvBuilder import EnvBuilder
from rl.environments.fl.BasicGridEnv import DrawInfo
from rl.environments.fl.FrozenLakeEnv import FrozenLakeEnv
from rl.tasks.fl.FLGrids import grid_map1, grid_map2, grid_map3, grid_map4

# ================================================ CONTROL PANEL =======================================================

# move, fall in hole, hit a wall, finish
rewards = [-1.0, -2.0, -2.0, 30.0]

color_map = [[0, 255, 255], [0, 0, 255], [0, 0, 0], [255, 255, 0], [0, 255, 0], [255, 0, 0]]

# Select grid (CNN - ~200+ for 1, ~4000+ for 2)
maps = {
    "grid_map1": grid_map1,
    "grid_map2": grid_map2,
    "grid_map3": grid_map3,
    "grid_map4": grid_map4
}

need_colorize_q_map = True

draw_info = DrawInfo.short

skip_frames = 30  # to speed up computations

colorize_q_map_frames_skip = 29  # to speed up computations

slow_render = 0  # to see what is going on more carefully

begin_from_start_if_get_in_hole = False

env_name = "FrozenLake"

map_name = "grid_map1"  # grid_map1, grid_map2, grid_map3, grid_map4

# Select agent
selected_agent = "TabTBQN"  # TabQ, TabDQ, CNNQ, CNNDQ, TabSARSA, CNNSARSA, TabESARSA, CNNESARSA, TabNSARSA, CNNNSARSA
# TabTBQN, CNNTBQN

iterations = 1000000

# ======================================================================================================================


class FrozenLakeMethod(Enum):
    TabQ = 0
    CNNQ = 1
    TabDQ = 2
    CNNDQ = 3
    TabSARSA =4
    CNNSARSA = 5
    TabESARSA = 6
    CNNESARSA = 7
    TabNSARSA = 8
    CNNNSARSA = 9
    TabTBQN = 10
    CNNTBQN = 11


def get_agent(method: FrozenLakeMethod, model_suffix, need_to_load):
    model_name = method.name + "_" + str(int(model_suffix))
    model_name = os.path.join(map_name, model_name)
    path = ProjectPath.join_to_table_models_path(os.path.join(env_name, model_name))
    path_nn = ProjectPath.join_to_nn_models_path(os.path.join(env_name, model_name))

    if method == FrozenLakeMethod.TabQ:
        return TabularQAgent()
    elif method == FrozenLakeMethod.CNNQ:
        return CNNQAgent(path_nn, load_model=need_to_load)
    elif method == FrozenLakeMethod.TabDQ:
        return TabularDoubleQAgent()
    elif method == FrozenLakeMethod.CNNDQ:
        return CNNDQAgent(path_nn, load_model=need_to_load)
    elif method == FrozenLakeMethod.TabSARSA:
        return TabularSARSAAgent()
    elif method == FrozenLakeMethod.CNNSARSA:
        return CNNSARSAAgent(path_nn, load_model=need_to_load)
    elif method == FrozenLakeMethod.TabESARSA:
        return TabularESARSAAgent()
    elif method == FrozenLakeMethod.CNNESARSA:
        return CNNESARSAAgent(path_nn, load_model=need_to_load)
    elif method == FrozenLakeMethod.TabNSARSA:
        return TabularNSARSAAgent()
    elif method == FrozenLakeMethod.CNNNSARSA:
        return CNNNSARSAAgent(path_nn, load_model=need_to_load)
    elif method == FrozenLakeMethod.TabTBQN:
        return TabularTreeBackupAgent()
    elif method == FrozenLakeMethod.CNNTBQN:
        return CNNTBQNAgent(path_nn, load_model=need_to_load)
    return None


class FrozenLakeEnvBuilder(EnvBuilder):

    def __init__(self, model_suffix, need_to_load, need_to_save, method):
        self._model_suffix = model_suffix
        self._need_to_load = need_to_load
        self._need_to_save = need_to_save
        self._method = method
        self._agent_name = selected_agent
        self._is_env_closed = None

    def episode_done(self, player_prop):
        if self._need_to_save:
            EnvBuilder.save_model(self._agent)

    def iteration_complete(self, state, action, reward, next_state, done, truncated, player_prop):
        pass

    def stop_render(self):
        return self._is_env_closed()

    def get_iterations(self):
        return iterations

    def build_env_and_agent(self) -> (Env, Dyna):
        grid_map = maps[map_name]
        env = FrozenLakeEnv(rewards=rewards, grid_map=grid_map[0], cell_width=grid_map[1], cell_height=grid_map[2],
                            color_map=color_map, begin_from_start_if_get_in_hole=begin_from_start_if_get_in_hole)
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
