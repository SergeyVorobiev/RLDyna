from enum import Enum

from rl.bonus.bandit import BanditMain
from rl.bonus.cartransfer import CarTransferMain
from rl.bonus.gridworld import GridWorldMain
from rl.tasks.EnvRendererInst import EnvRendererInst
from rl.environments.EnvBuilder import EnvBuilder
from rl.environments.fl.FrozenLakeEnvBuilder import FrozenLakeMethod, FrozenLakeEnvBuilder
from rl.environments.mountaincar.MountainCarEnvBuilder import MountainCarMethod, MountainCarEnvBuilder
from rl.environments.cartpole.CartPoleEnvBuilder import CartPoleMethod, CartPoleEnvBuilder
from rl.environments.shelter.ShelterEnvBuilder import ShelterMethod, ShelterEnvBuilder
from rl.environments.cliffwalking.CliffWalkingEnvBuilder import CliffWalkingMethod, CliffWalkingEnvBuilder
from rl.environments.lunarlander.LunarLanderEnvBuilder import LunarLanderMethod, LunarLanderEnvBuilder
from rl.environments.lunarlandercont.LunarLanderContEnvBuilder import LunarLanderContMethod, LunarLanderContEnvBuilder


# When the model is saved it has two indices:
# 1 is model_name_suffix and can be edited from GameNameEnvBuilder
# 2 is model_index, by default 0, represent model index in array, and can be edited in appropriate model classes as
# model_index

# Go into appropriate classes to do setup in Control Panel.

# GYM VERSION IS 21, 26:
# 1 HAS SLOWER VISUAL RENDERING
# 2 HAS PROBLEMS WITH ATARI
class Game(Enum):
    Bandit = -1
    CarTransfer = -2
    GridWorld = - 3

    ShowGame = 0
    FrozenLake = 1
    MountainCar = 2
    CartPole = 3
    Shelter = 4
    CliffWalking = 5
    LunarLander = 6
    LunarLanderCont = 7


# Env Builder, model suffix to save, need to load model, need to save model, method
builders = {

    "FrozenLake": ["FrozenLakeEnvBuilder", 1, True, True, FrozenLakeMethod.TabTBQN],

    "MountainCar": ["MountainCarEnvBuilder", 1, True, True, MountainCarMethod.NNSARSALambda],

    "CartPole": ["CartPoleEnvBuilder", 1, True, True, CartPoleMethod.TDPGDLAC],

    # QAlgorithm and MCPGDAlgorithm (that is not intended for this task)
    "Shelter": ["ShelterEnvBuilder", 1, True, True, ShelterMethod.Q],

    # See the difference between SARSAAlgorithm & QAlgorithm
    "CliffWalking": ["CliffWalkingEnvBuilder", 1, True, True, CliffWalkingMethod.SARSA],

    "LunarLander": ["LunarLanderEnvBuilder", 1, True, True, LunarLanderMethod.MCPGDL],

    "LunarLanderCont": ["LunarLanderContEnvBuilder", 1, True, True, LunarLanderContMethod.TDPGCAC],

    # "BipedalWalker-v3", "LunarLander-v2", "CarRacing-v0", "Pendulum-v1", etc.
    # "AirRaid-v0", "SpaceInvaders-v0", "MsPacman-v4", ALE/Backgammon-v5 - Could require Atari emulator,
    # below commands could help:
    # pip install ale-py
    # pip install gym[accept-rom-license]

    # Game name, print info, delay_frame
    "ShowGame": ["EmptyEnvBuilder", "MsPacman-v4", True, 0]
}


def build_env() -> EnvBuilder:
    if builder.name == "ShowGame":
        li = builders[builder.name]
        return globals()[li[0]](li[1], li[2], li[3])
    else:
        li = builders[builder.name]
        return globals()[li[0]](li[1], li[2], li[3], li[4])


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

builder = Game.LunarLander

# **********************************************************************************************************************

# tensorflow 2.12.0
# keras 2.12.0
# gym 0.21.0
# matplotlib 3.7.1
if __name__ == '__main__':
    if builder == builder.Bandit:
        BanditMain.run()
    elif builder == builder.GridWorld:
        GridWorldMain.run()
    elif builder == builder.CarTransfer:
        CarTransferMain.run()
    else:
        EnvRendererInst(build_env()).render()
