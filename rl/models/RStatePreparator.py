from abc import abstractmethod


class RStatePreparator:

    @abstractmethod
    def prepare_raw_state(self, raw_state):
        ...
