from rl.environments.cartpole.CartPoleEnvBuilder import CartPoleEnvBuilder
from rl.environments.EnvBuilder import EnvBuilder
from rl.environments.fl.FLEnvBuilder import FLEnvBuilder
from rl.environments.mountaincar.MountainCarEnvBuilder import MountainCarEnvBuilder
from rl.environments.EmptyEnvBuilder import EmptyEnvBuilder
from rl.tasks.EnvRendererInst import EnvRendererInst

# Go into appropriate classes to do setup in Control Panel.
builders = {

    "FrozenLake": "FLEnvBuilder",

    # 500 - 1000 episodes to see the effect
    "MountainCar": "MountainCarEnvBuilder",

    # The goal is to hold up for 500 iterations then the episode naturally ends, 10000+ episodes to learn.
    "CartPole": "CartPoleEnvBuilder",

    # "BipedalWalker-v3", "LunarLander-v2", "CarRacing-v0" etc.
    # "AirRaid-v0", "SpaceInvaders-v0", "MsPacman-v0" - Could require Atari emulator, below commands could help:
    # pip install ale-py
    # pip install gym[accept-rom-license]

    # Game name, print info, delay_frame
    "ShowGame": ["EmptyEnvBuilder", "BipedalWalker-v3", True, 0]
}

builder = "MountainCar"  # FrozenLake, CartPole, MountainCar, ShowGame


def build_env() -> EnvBuilder:
    if builder == "ShowGame":
        li = builders[builder]
        return globals()[li[0]](li[1], li[2], li[3])
    else:
        return globals()[builders[builder]]()


if __name__ == '__main__':
    EnvRendererInst(build_env()).render()
