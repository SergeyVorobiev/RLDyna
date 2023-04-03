from rl.stateprepares.RStatePrepare import RStatePrepare


class CartPoleDiscreteStatePrepare(RStatePrepare):

    def __init__(self, digitizer):
        self._digitizer = digitizer

    def prepare_raw_state(self, raw_state):
        return self._digitizer(raw_state)
