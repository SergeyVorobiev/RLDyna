from collections import deque


# Keep the state its q and a in map, deque guarantee the order by FIFO and size limit.
# Keep in mind that hash does not guarantee the uniqueness of the state (i.e. potentially two different states can have
# the same hash)
from typing import Any


# Sometimes it is very convenient to keep only the unique states
class StateHashBank:

    def __init__(self, max_len: int):
        self._deque = deque(maxlen=max_len)
        self._dict = dict()

    # Just rewrite the state by its hash but keep counting
    def set(self, hash_value: int, state: Any, qs: [float], action: int):
        value = self._dict.get(hash_value)
        qs_copy = qs.copy()
        if value is None:
            counter = [0] * qs.__len__()
            counter[action] = 1
            self._dict[hash_value] = [qs_copy, state, counter]
        else:
            counter = value[2]
            counter[action] = counter[action] + 1
            self._dict[hash_value] = [qs_copy, state, counter]

    # Update the state and qs for the specified hash, if there was no such hash, the new will be added by using
    # default_qs, the default_qs will also be updated by using q and a
    def update(self, hash_value: int, state: Any, q: float, action: int, default_qs: [float]) -> [int, Any]:
        value = self._dict.get(hash_value)
        if value is None:
            default_qs = default_qs.copy()
            default_qs[action] = q
            counter = [0] * default_qs.__len__()
            counter[action] = 1
            value = [default_qs, state, counter]
            self._dict[hash_value] = value
            if self._deque.__len__() == self._deque.maxlen:
                old_value = self._deque.popleft()
                self._dict.pop(old_value)
            self._deque.append(hash_value)
            return value
        else:
            qs = value[0]
            counter = value[2]
            counter[action] = counter[action] + 1
            qs[action] = q
            self._dict[hash_value] = [qs, value[1], counter]
        return value

    # return all qs for all states and counters. Counters contain the numbers of usages of the particular state-actions.
    def get_values(self):
        qs = []
        states = []
        counters = []
        for pair in self._dict.items():
            qs.append(pair[1][0])
            states.append(pair[1][1])
            counters.append(pair[1][2])
        return qs, states, counters

    # return all qs, state, counters for hash. Counters contain the numbers of usages of the particular state-actions.
    def get(self, hash_value: int) -> [float, Any, int] or None:
        try:
            return self._dict[hash_value]
        except Exception as e:
            return None

    def clear(self):
        self._deque.clear()
        self._dict.clear()

    def has_hash(self, hash_value):
        return self._dict.__contains__(hash_value)

    def __str__(self):
        return self._deque.__str__()