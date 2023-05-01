from abc import abstractmethod

from rl.mdp.Action import Action
from rl.mdp.State import State


class Transition(object):

    @abstractmethod
    def get_transition(self, s: State, a: Action, sn: State) -> (float, float):
        ...
