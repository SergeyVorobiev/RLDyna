from abc import abstractmethod

from rl.mdp.Action import Action


class Visual(object):

    def __init__(self):
        pass

    @abstractmethod
    def set_actions(self, actions: [Action]):
        ...

    @abstractmethod
    def set_optimal_actions_count(self, optimal_actions_count: int):
        ...

    @abstractmethod
    def set_state_v(self, v: float):
        ...

    @abstractmethod
    def set_state_name(self, state_name: str):
        ...

    @abstractmethod
    def set_optimal_q(self, optimal_q: float):
        ...
