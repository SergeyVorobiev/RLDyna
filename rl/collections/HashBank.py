from collections import deque
from typing import Any


# Keeps only objects with unique hash. Deque is used to keep order and max_len (FIFO)
class HashBank:

    def __init__(self, max_len: int):
        self._deque = deque(maxlen=max_len)
        self._dict = dict()

    # Update the value if hash exists
    def add(self, hash_value: int, obj: Any):
        if hash_value not in self._dict:
            self._dict[hash_value] = obj
            if self._deque.__len__() == self._deque.maxlen:
                old_value = self._deque.popleft()
                self._dict.pop(old_value)
            self._deque.append(hash_value)
        else:
            self._dict[hash_value] = obj

    def __len__(self):
        return self._deque.__len__()

    def get_values(self):
        return self._dict.values()

    def get_items(self):
        items = []
        for item in self._dict.items():
            items.append(item)
        return items

    def clear(self):
        self._deque.clear()
        self._dict.clear()
