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
    def iteration_complete(self, state, action, reward, next_state, done, truncated, player_prop):
        ...

    @abstractmethod
    def stop_render(self):
        ...

    @staticmethod
    def save_model(agent):
        if agent is not None:
            models = agent.get_models()
            for model in models:
                result, saved_path = model.save()
                if result:
                    print("Model is saved: " + saved_path)
