from rl.environments.cartpole.CartPoleEnvBuilder import CartPoleEnvBuilder
from rl.environments.EnvBuilder import EnvBuilder
from rl.environments.fl.FLEnvBuilder import FLEnvBuilder
from rl.tasks.EnvRenderer import EnvRenderer


# Go into appropriate classes to do setup in Control Panel.
builders = {

    "FrozenLake": "FLEnvBuilder",

    "CartPole": "CartPoleEnvBuilder"
}

builder = "CartPole"  # FrozenLake, CartPole


if __name__ == '__main__':
    env_builder: EnvBuilder = globals()[builders[builder]]()

    env, agent = env_builder.build_env_and_agent()

    EnvRenderer.render(env, agent, env_builder.get_iterations(), episode_done_listener=env_builder.episode_done,
                       iteration_complete_listener=env_builder.iteration_complete,
                       stop_render_listener=env_builder.stop_render)
