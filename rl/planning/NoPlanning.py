from rl.algorithms.RAlgorithm import RAlgorithm
from rl.models import RModel
from rl.planning.RPlanning import RPlanning


class NoPlanning(RPlanning):
    def plan(self, models: [RModel], algorithm: RAlgorithm):
        pass
