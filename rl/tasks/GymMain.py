from rl.environments.cartpole.CartPoleEnvBuilder import CartPoleEnvBuilder
from rl.environments.EnvBuilder import EnvBuilder
from rl.environments.fl.FLEnvBuilder import FLEnvBuilder
from rl.tasks.EnvRendererInst import EnvRendererInst

# Go into appropriate classes to do setup in Control Panel.
builders = {

    "FrozenLake": "FLEnvBuilder",

    "CartPole": "CartPoleEnvBuilder"
}

builder = "CartPole"  # FrozenLake, CartPole

if __name__ == '__main__':
    env_builder: EnvBuilder = globals()[builders[builder]]()
    EnvRendererInst(env_builder).render()
