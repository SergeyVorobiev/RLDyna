import random
from typing import Any

from rl.models.RModel import RModel
from rl.algorithms.RAlgorithm import RAlgorithm
from rl.planning.RPlanning import RPlanning


class SimplePlanning(RPlanning):

    def __init__(self, plan_batch_size: int, plan_step_size: int, memory_size: int):
        super().__init__(memory_size)
        self._plan_batch_size: int = plan_batch_size
        self.__error = 0
        self.__error2 = 0
        self._epsilon: float = 0
        self._plan_steps = 0
        self._plan_steps_size = plan_step_size

    def memorize(self, batch: Any):
        if batch[0] > self.__error:
            self._memory.append(batch)
        else:
            if random.uniform(0, 1) < self._epsilon:
                self._memory.append(batch)

    def plan(self, models: [RModel], algorithm: RAlgorithm):
        self._plan_steps += 1
        if self._plan_steps == self._plan_steps_size:
            self._plan_steps = 0
            if self._memory.__len__() >= self._plan_batch_size:
                batch = random.sample(self._memory, self._plan_batch_size)
                for error, sample in batch:
                    _ = algorithm.train_from_past(models, sample)
                    # print(error)
                    # self._memory.append((error, sample))
                    # while error > self.__error2:
                    #    error, _ = algorithm.train_batch(models, policy, sample)
            # last = self.get_last_memorized()
            # if last is not None and last.__len__() > 0:
                # error = self.get_last_memorized()[0][0]
                # if error < self.__error:
                    # print("Remove")
                    # self.remove_last()
