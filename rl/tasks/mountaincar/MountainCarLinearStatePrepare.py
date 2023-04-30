import numpy

from rl.stateprepares.RStatePrepare import RStatePrepare


class MountainCarLinearStatePrepare(RStatePrepare):

    def prepare_raw_state(self, raw_state):
        s1 = raw_state[0]
        s2 = raw_state[1]
        s1s2 = s1 * s2
        s12 = s1 ** 2
        s22 = s2 ** 2
        s1s22 = s1 * s22
        s12s2 = s12 * s2
        s12s22 = s12 * s22
        return numpy.array([1, s1, s2, s1s2, s12, s22, s1s22, s12s2, s12s22])
