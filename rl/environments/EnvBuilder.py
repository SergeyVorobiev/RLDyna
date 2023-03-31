from abc import abstractmethod
from typing import Any

from gym import Env

from rl.dyna.Dyna import Dyna


class EnvBuilder(object):

    @abstractmethod
    def build_env_and_agent(self) -> (Env, Dyna):
        ...

    @abstractmethod
    def get_iterations(self):
        ...

    @abstractmethod
    def episode_done(self, player_prop: Any):
        ...

    @abstractmethod
    def iteration_complete(self, player_prop: Any):
        ...

    @abstractmethod
    def stop_render(self):
        ...

    @abstractmethod
    def lookup_listener(self, state, action, reward, next_state, done, player_prop):
        ...

