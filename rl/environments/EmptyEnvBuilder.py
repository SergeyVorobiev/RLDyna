import time
from typing import Any

from gym import Env

from rl.agents.EmptyAgent import EmptyAgent
from rl.dyna.Dyna import Dyna
from rl.environments.EnvBuilder import EnvBuilder
import gym


# ================================================ CONTROL PANEL =======================================================

iterations = 1000000

# ======================================================================================================================


class EmptyEnvBuilder(EnvBuilder):

    def __init__(self, game_name, print_state_info=True, delay_frame=0):
        self._game_name = game_name
        self._ep_iter = 0
        self._episodes = 0
        self._print_state_info = print_state_info
        self._delay_frame = delay_frame
        self._env = None
        self._agent = None

    def get_iterations(self):
        return iterations

    def episode_done(self, player_prop: Any):
        self._episodes += 1
        print(f"Episode {self._episodes}: Iterations: {self._ep_iter}")
        self._ep_iter = 0

    def iteration_complete(self, state, action, reward, next_state, done, truncated, player_prop):
        self._ep_iter += 1
        if self._print_state_info:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print(f"Action space: {self._env.action_space}\n")
            print(f"Env space: {self._env.observation_space}\n")
            print("===============State======================================================")
            print(state)
            print("==========================================================================\n")
            print(f"Action: {action}   Reward: {reward}   Done: {done}\n")
            print(f"Player prop: {player_prop}\n")
            print("===============Next State=================================================")
            print(next_state)
            print("==========================================================================")
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")
        time.sleep(self._delay_frame)

    def stop_render(self):
        pass

    def build_env_and_agent(self) -> (Env, Dyna):
        self._env: Env = gym.make(self._game_name)
        self._agent = EmptyAgent().build_agent(self._env)
        return self._env, self._agent
