import numpy as np

from rl.stateprepares.RStatePrepare import RStatePrepare


# Allows convert or modify every row state of environment before performing the calculations.
class ShelterStateNNPrepare(RStatePrepare):

    def __init__(self):
        self._divider = 50 / 255

    def prepare_raw_state(self, raw_state):
        raw_state = raw_state * self._divider

        # (8, 7, 1)
        return np.expand_dims(raw_state, axis=-1)
