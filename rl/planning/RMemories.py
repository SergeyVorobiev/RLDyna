from abc import abstractmethod
from typing import Any


class RMemories(object):

    # returns - state, action, reward, next_state, done, env_props
    @abstractmethod
    def get_last_memorized(self, size: int = 1, last_first=True) -> [(Any, int, float, Any, bool, Any)]:
        ...
