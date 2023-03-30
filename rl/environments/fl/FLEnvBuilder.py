import os

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
rewards = [-1, -5, -3, 30]

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

load_model = True
need_to_save_model = True

env_name = "FrozenLake"

map_name = "grid_map1"  # grid_map1, grid_map2, grid_map3, grid_map4

path = ProjectPath.join_to_nn_models_path(os.path.join(env_name, map_name))

agents = {
    "TabQ": TabularQAgent(),
    "CNNQ": CNNQAgent(os.path.join(path, "CNNQ"), load_model=load_model),
    "TabDQ": TabularDoubleQAgent(),
    "CNNDQ": CNNDQAgent(os.path.join(path, "CNNDQ"), load_model=load_model),
    "TabSARSA": TabularSARSAAgent(),
    "CNNSARSA": CNNSARSAAgent(os.path.join(path, "CNNSARSA"), load_model=load_model),
    "TabESARSA": TabularESARSAAgent(),
    "CNNESARSA": CNNESARSAAgent(os.path.join(path, "CNNESARSA"), load_model=load_model),
    "TabNSARSA": TabularNSARSAAgent(),
    "CNNNSARSA": CNNNSARSAAgent(os.path.join(path, "CNNNSARSA"), load_model=load_model),
    "TabTBQN": TabularTreeBackupAgent(),
    "CNNTBQN": CNNTBQNAgent(os.path.join(path, "CNNTBQN"), load_model=load_model)
}

# Select agent
selected_agent = "TabTBQN"  # TabQ, TabDQ, CNNQ, CNNDQ, TabSARSA, CNNSARSA, TabESARSA, CNNESARSA, TabNSARSA, CNNNSARSA
# TabTBQN, CNNTBQN

iterations = 1000000

# ======================================================================================================================


class FLEnvBuilder(EnvBuilder):

    def __init__(self):
        self._agent_name = selected_agent
        self._is_env_closed = None

    def episode_done(self, player_prop):
        if need_to_save_model and self._agent is not None:
            if self._agent.get_models()[0].save():
                print("Model is saved")

    def iteration_complete(self, player_prop):
        pass

    def stop_render(self):
        return self._is_env_closed()

    def get_iterations(self):
        return iterations

    def build_env_and_agent(self) -> (Env, Dyna):
        grid_map = maps[map_name]
        env = FrozenLakeEnv(rewards=rewards, grid_map=grid_map[0], cell_width=grid_map[1], cell_height=grid_map[2],
                            color_map=color_map, begin_from_start_if_get_in_hole=False)
        self._is_env_closed = env.is_closed

        # Skip some part of screen updating to speed up the process.
        env.set_skip_frame(skip_frames)
        env.draw_map_frame_skip(colorize_q_map_frames_skip)
        env.set_slow_render(slow_render)
        env.draw_info(draw_info)
        env.draw_map(draw_map=need_colorize_q_map)

        # build Dyna agent
        self._agent = agents[self._agent_name].build_agent(env)
        env.draw_values_setup(q_supplier=self._agent.get_q_values)

        return env, self._agent
