from abc import abstractmethod
from typing import Any

from rl.planning.RMemories import RMemories


class RMemory(RMemories):

    # state: Any, action: int, reward: float, next_state: Any, done: bool, props: Any
    @abstractmethod
    def memorize(self, batch):
        ...

    @abstractmethod
    def get_memory_size(self) -> int:
        ...

    @abstractmethod
    def is_memory_full(self) -> bool:
        ...

    @abstractmethod
    def get_memory_capacity(self) -> int:
        ...

    @abstractmethod
    def clear_memory(self) -> Any:
        ...
