from abc import abstractmethod

from gym import Env

from rl.dyna.Dyna import Dyna


class RDynaAgentBuilder(object):

    @abstractmethod
    def build_agent(self, env: Env) -> Dyna:
        ...
