from rl.helpers.ImageHelper import save_input_as_image
from rl.models.stateprepares.RStatePrepare import RStatePrepare
import numpy as np


# Allows convert or modify every row state of environment before performing the calculations.
class FrozenLakeCNNStatePrepare(RStatePrepare):

    def __init__(self):
        self._divider = 50 / 255
        pass

    def prepare_raw_state(self, raw_state):

        # If you use NN model the state could be transformed to some ndarray or be prepared (grey colored etc) here.
        raw_state1 = raw_state[0]
        raw_state2 = raw_state[1]
        raw_state1 = raw_state1 * self._divider
        raw_state2 = raw_state2 * self._divider
        # save_input_as_image(raw_state, "PreparedState")  # save image to see resulted state for NN in resources folder
        raw_state1 = np.expand_dims(raw_state1, axis=-1)
        raw_state2 = np.expand_dims(raw_state2, axis=-1)
        return [raw_state1, raw_state2]
