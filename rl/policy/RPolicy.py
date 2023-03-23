from abc import abstractmethod


class RPolicy(object):

    # Returns number of action
    @abstractmethod
    def pick(self, values) -> int:
        ...

    @abstractmethod
    def get_action_probability(self, values, action) -> float:
        ...

    @abstractmethod
    def improve(self):
        ...
