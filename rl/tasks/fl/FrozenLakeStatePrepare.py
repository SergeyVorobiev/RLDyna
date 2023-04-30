from rl.stateprepares.RStatePrepare import RStatePrepare


# Allows convert or modify every row state of environment before performing the calculations.
class FrozenLakeStatePrepare(RStatePrepare):

    def __init__(self, y):
        self._y = y

    def prepare_raw_state(self, raw_state):

        # So as the current Frozen lake is a grid X x Y, and we use tabular approach,
        # we transform the state into the single number to make an easy index for the data table

        # If you use NN model the state could be transformed to some ndarray or be prepared (grey colored etc) here.
        return raw_state[0] * self._y + raw_state[1]
