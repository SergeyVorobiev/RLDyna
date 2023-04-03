from rl.stateprepares.RStatePrepare import RStatePrepare


# Allows convert or modify every row state of environment before performing the calculations.
class ShelterStatePrepare(RStatePrepare):

    def __init__(self, y):
        self._y = y

    def prepare_raw_state(self, raw_state):
        return raw_state[0] * self._y + raw_state[1]
