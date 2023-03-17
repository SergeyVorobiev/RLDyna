from typing import Union

from rl.visual.Visual import Visual
from rl.mdp.Action import Action


class State(object):

    def __init__(self, state_id: int, name: str, actions: {object: Union[Action, str, int]} or [Union[Action, str, int]], info=None, visual=None):
        self.state_id: int = state_id
        self.name: str = name
        self.v: float = 0
        self.visit_count: int = 0
        self.__actions: {object: Action}
        self.__actions = State.__make_copy(actions)
        self.optimal_actions_count = 0
        self.probability = 0
        self.visited = False
        self.optimal_actions: [Action] = []
        self.max_q = None
        self.action_size = self.__actions.__len__()
        self.q_values = [0] * self.action_size
        for action in self.__actions.values():
            self.optimal_actions.append(action)
        self.info = info
        self.visual: Visual = visual
        if self.visual is not None:
            self.visual.set_state_name(self.name)

    # return list cached in state, do not change it.
    def update_and_get_q_values(self):
        for i in range(self.action_size):
            self.q_values[i] = self.optimal_actions[i].q
        return self.q_values

    def get_max_q_value(self):
        max_value = self.optimal_actions[0].q
        for action in self.optimal_actions:
            if max_value < action.q:
                max_value = action.q
        return max_value

    def get_max_a(self):
        max_value = self.optimal_actions[0].q
        a_i = 0
        i = 0
        for action in self.optimal_actions:
            if max_value < action.q:
                a_i = i
                max_value = action.q
            i += 1
        return a_i

    def get_action_by_key(self, key):
        return self.__actions[key]

    def set_actions(self, actions: {object: Union[Action, str, int]} or [Union[Action, str, int]], copy=True):
        if copy:
            self.__actions = State.__make_copy(actions)
        else:
            self.__actions = actions

    @staticmethod
    def __make_copy(actions: {object: Action} or [Action]) -> {object: Action}:
        if actions is None:
            return None
        new_actions = {}
        if isinstance(actions, dict):
            actions_to_copy = actions.values()
        else:
            actions_to_copy = actions
        k = 0
        for action in actions_to_copy:
            if type(action) == str:
                action = Action(k, action)
            elif type(action) == int:
                action = Action(k, action)
            elif type(action) != Action:
                raise Exception("Not supported type of action, should be: str/int/Action")
            new_actions[action.key] = action.default_clone(k)
            k += 1
        return new_actions

    def get_actions(self):
        return self.__actions.values()

    # create new list on each invoke
    def get_optimal_actions(self):
        result = []
        for i in range(self.optimal_actions_count):
            result.append(self.optimal_actions[i])
        return result

    def get_actions_dict(self) -> {object: Action}:
        return self.__actions

    def __str__(self):
        return "id: " + str(self.state_id) + " name: " + str(self.name)

    def update_visual(self):
        if self.visual is not None:
            self.visual.set_actions(self.optimal_actions)
            self.visual.set_state_v(self.v)
            self.visual.set_optimal_actions_count(self.optimal_actions_count)
            self.visual.set_optimal_q(self.max_q)

