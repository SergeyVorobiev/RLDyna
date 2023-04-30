from typing import Any

from rl.collections.HashBank import HashBank
from rl.models.RModel import RModel
from rl.algorithms.RAlgorithm import RAlgorithm
from rl.planning.RPlanning import RPlanning


class HashPlanning(RPlanning):

    def __init__(self, plan_batch_size: int, plan_step_size: int, memory_size: int):
        super().__init__(memory_size)
        self._plan_batch_size: int = plan_batch_size
        self._plan_steps_size = plan_step_size
        self._plan_steps = 0
        self._hash_bank = HashBank(plan_batch_size)

    def memorize(self, batch: Any):
        self._memory.append(batch)

    def plan(self, models: [RModel], algorithm: RAlgorithm):
        self._plan_steps += 1
        model: RModel = models[0]
        if self._plan_steps == self._plan_steps_size:
            self._plan_steps = 0

            # Just get all unique from memory
            for batch in self._memory:
                state, action, reward, next_state, done, _, _ = batch[1][0]
                hash_value = model.get_state_hash(state)
                self._hash_bank.add(hash_value, batch)
                if self._hash_bank.__len__() == self._plan_batch_size:
                    break
            values = self._hash_bank.get_values()
            for error, sample in values:
                _ = algorithm.plan(models, sample)
