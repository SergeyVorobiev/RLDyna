from abc import abstractmethod


class RStatePrepare:

    @abstractmethod
    def prepare_raw_state(self, raw_state):
        ...
