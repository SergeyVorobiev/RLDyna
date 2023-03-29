import random
from typing import Any

from rl.models.RModel import RModel
from rl.algorithms.RAlgorithm import RAlgorithm
from rl.planning.RPlanning import RPlanning


class SimplePlanning(RPlanning):

    def __init__(self, plan_batch_size: int, plan_step_size: int, memory_size: int, clear_memory_after_planning=False):
        super().__init__(memory_size)
        self._plan_batch_size: int = plan_batch_size
        self._plan_steps = 0
        self._plan_steps_size = plan_step_size
        self._clear_memory_after_planning = clear_memory_after_planning

    def memorize(self, batch: Any):
        self._memory.append(batch)

    def plan(self, models: [RModel], algorithm: RAlgorithm):
        self._plan_steps += 1
        if self._plan_steps == self._plan_steps_size:
            self._plan_steps = 0
            if self._memory.__len__() >= self._plan_batch_size:
                batch = random.sample(self._memory, self._plan_batch_size)
                for error, sample in batch:
                    _ = algorithm.plan(models, sample)
            if self._clear_memory_after_planning:
                self.clear_memory()
