from rl.ProjectPath import ProjectPath
from rl.agents.CNNQAgent import CNNQAgent
from rl.agents.CNNTBQNAgent import CNNTBQNAgent
from rl.agents.TabularQAgent import TabularQAgent
from rl.agents.TabularTreeBackup import TabularTreeBackup
from rl.dyna.Dyna import Dyna
from rl.environment.BasicGridEnv import DrawInfo
from rl.environment.FrozenLakeEnv import FrozenLakeEnv
from rl.tasks.EnvRenderer import EnvRenderer
from rl.tasks.fl.FLGrids import grid_map1, grid_map2, grid_map3, grid_map4

# ================================================ CONTROL PANEL ======================================================

agents = {
    "TabQ": TabularQAgent(),
    "CNNQ": CNNQAgent(ProjectPath.join_to_res_models_path("CNNQ"), load_model=True),
    "TabTBQN": TabularTreeBackup(),
    "CNNTBQN": CNNTBQNAgent(ProjectPath.join_to_res_models_path("CNNTBQN"), load_model=True)
}

# move, fall in hole, hit a wall, finish
rewards = [-1, -5, -3, 30]

color_map = [[0, 255, 255], [0, 0, 255], [0, 0, 0], [255, 255, 0], [0, 255, 0], [255, 0, 0]]

# Select grid (CNNTBQN - ~200+ for 1, ~4000+ for 2)
grid_map = grid_map1  # 4, 3, 2, 1

env = FrozenLakeEnv(rewards=rewards, grid_map=grid_map[0], cell_width=grid_map[1], cell_height=grid_map[2],
                    color_map=color_map, begin_from_start_if_get_in_hole=False)

# Select agent
selected_agent = "CNNQ"  # TabQ, CNNQ, TabTBQN, CNNTBQN

iterations = 1000000

need_colorize_q_map = True

draw_info = DrawInfo.short

skip_frames = 30  # to speed up computations

colorize_q_map_frames_skip = 29  # to speed up computations

slow_render = 0  # to see what is going on more carefully

# =====================================================================================================================


def setup_env():
    # Skip some part of screen updating to speed up the process.
    env.set_skip_frame(skip_frames)
    env.draw_map_frame_skip(colorize_q_map_frames_skip)
    env.set_slow_render(slow_render)

    env.draw_info(draw_info)
    env.draw_values_setup(q_supplier=agent.get_q_values, draw_map=need_colorize_q_map)


def episode_end():

    # Save the model after each episode
    agent.get_models()[0].save()


def is_stop():

    # Stop the process if the window is off
    return env.is_closed()


if __name__ == '__main__':
    # build Dyna agent
    agent: Dyna = agents[selected_agent].build_agent(env)

    setup_env()
    EnvRenderer.render(env, agent, iterations, episode_done_listener=episode_end, stop_listener=is_stop)
