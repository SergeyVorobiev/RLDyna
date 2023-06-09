from collections import deque
from typing import Any

from rl.planning.RMemory import RMemory


class RBaseMemory(RMemory):

    def __init__(self, memory_capacity: int = 1):
        self.memory_capacity: int = memory_capacity
        self._memory: deque = deque(maxlen=memory_capacity)

    def is_memory_full(self):
        return self._memory.__len__() == self.memory_capacity

    def clear_memory(self) -> Any:
        self._memory.clear()

    def get_memory_capacity(self):
        return self.memory_capacity

    def get_memory_size(self):
        return self._memory.__len__()

    def remove_last_memories(self, size: int = 1):
        if size >= self.get_memory_size():
            self.clear_memory()
        else:
            for _ in range(size):
                _ = self._memory.pop()

    # state: Any, action: int, reward: float, next_state: Any, done: bool, props: Any
    def get_last_memorized(self, size: int = 1, last_first=True) -> [Any]:
        count = 0
        result = []
        if last_first:
            for sample in reversed(self._memory):
                count += 1
                if count > size:
                    break
                result.append(sample)
        else:
            memory_size = self._memory.__len__()
            start = memory_size - size
            if start < 0:
                start = 0
            for i in range(start, memory_size):
                result.append(self._memory[i])
        return result

    def memorize(self, batch: Any):
        self._memory.append(batch)
