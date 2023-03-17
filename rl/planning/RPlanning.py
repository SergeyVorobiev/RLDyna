from abc import abstractmethod

from rl.models import RModel
from rl.algorithms.RAlgorithm import RAlgorithm
from rl.planning.RBaseMemory import RBaseMemory


class RPlanning(RBaseMemory):

    @abstractmethod
    def plan(self, models: [RModel], algorithm: RAlgorithm):
        ...
