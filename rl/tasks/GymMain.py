from rl.tasks.EnvRendererInst import EnvRendererInst
from rl.environments.EnvBuilder import EnvBuilder
from rl.environments.fl.FrozenLakeEnvBuilder import FrozenLakeEnvBuilder
from rl.environments.mountaincar.MountainCarEnvBuilder import MountainCarEnvBuilder
from rl.environments.cartpole.CartPoleEnvBuilder import CartPoleEnvBuilder
from rl.environments.shelter.ShelterEnvBuilder import ShelterEnvBuilder
from rl.environments.cliffwalking.CliffWalkingEnvBuilder import CliffWalkingEnvBuilder
from rl.environments.EmptyEnvBuilder import EmptyEnvBuilder


# When the model is saved it has two indices:
# 1 is model_name_suffix and can be edited from GameNameEnvBuilder
# 2 is model_index, by default 0, represent model index in array, and can be edited in appropriate model classes as
# model_index

# Go into appropriate classes to do setup in Control Panel.
builders = {

    "FrozenLake": "FrozenLakeEnvBuilder",

    # 500 - 1000 episodes to see the effect
    "MountainCar": "MountainCarEnvBuilder",

    # The goal is to hold up for 500 iterations then the episode naturally ends, 10000+ episodes to learn.
    "CartPole": "CartPoleEnvBuilder",

    # QAlgorithm and MCPGAverBaselineAlgorithm (that is not intended for this task)
    "Shelter": "ShelterEnvBuilder",

    # See the difference between SARSAAlgorithm & QAlgorithm, QAlgorithm runs by default, to set up,
    # go to cw env builder
    "CliffWalking": "CliffWalkingEnvBuilder",

    # "BipedalWalker-v3", "LunarLander-v2", "CarRacing-v0", "Pendulum-v1", etc.
    # "AirRaid-v0", "SpaceInvaders-v0", "MsPacman-v0", ALE/Backgammon-v5 - Could require Atari emulator,
    # below commands could help:
    # pip install ale-py
    # pip install gym[accept-rom-license]

    # Game name, print info, delay_frame
    "ShowGame": ["EmptyEnvBuilder", "BipedalWalker-v3", True, 0]
}

builder = "CartPole"  # FrozenLake, CartPole, MountainCar, Shelter, CliffWalking, ShowGame


def build_env() -> EnvBuilder:
    if builder == "ShowGame":
        li = builders[builder]
        return globals()[li[0]](li[1], li[2], li[3])
    else:
        return globals()[builders[builder]]()


if __name__ == '__main__':
    EnvRendererInst(build_env()).render()
