from typing import Any

from rl.mdp.State import State
from rl.models.RTableModel import RTableModel


class Table(object):

    def __init__(self, n_states, n_actions):
        actions = []
        self.states_table = []
        for i in range(n_actions):
            actions.append(i)
        for i in range(n_states):
            state = State(i, str(i), actions)
            self.states_table.append(state)


class TableSingle(RTableModel):

    def save(self, path=None):
        pass

    def __init__(self, n_states: int, n_actions: int):
        super().__init__(n_actions)
        self._models = []
        self._models.append(Table(n_states, n_actions))

    def get_state_hash(self, state) -> Any:
        return hash(state.data.tobytes())

    def reset_actions_visit(self, model_index: int = 0):
        for state in self._models[model_index].states_table:
            for action in state.optimal_actions:
                action.visited = False

    def is_action_visited(self, state, action, model_index: int = 0):
        state_wrapper: State = self.__get_state(state, model_index)
        return state_wrapper.optimal_actions[action].visited

    def get_action_visit_count(self, state: Any, action: int, model_index: int = 0):
        state_wrapper: State = self.__get_state(state, model_index)
        return state_wrapper.optimal_actions[action].visit_count

    def get_state_visit_count(self, state: Any, model_index: int = 0):
        return self.__get_state(state, model_index).visit_count

    def update_state_visit_count(self, state: Any, model_index: int = 0):
        state_wrapper = self.__get_state(state, model_index)
        state_wrapper.visit_count += 1
        state_wrapper.visited = True

    def update_action_visit_count(self, state: Any, action: int, model_index: int = 0):
        action_wrapper = self.__get_state(state, model_index)
        action_wrapper.visit_count += 1
        action_wrapper.visited = True

    def get_max_a(self, state: Any, model_index: int = 0):
        state_wrapper: State = self.__get_state(state, model_index)
        return state_wrapper.get_max_a()

    # Be careful, it is returned original array of values for increasing speed.
    def get_q_values(self, state: Any, model_index: int = 0):
        state_wrapper: State = self.__get_state(state, model_index)
        return state_wrapper.update_and_get_q_values()

    def get_max_q(self, state: Any, model_index: int = 0):
        state_wrapper: State = self.__get_state(state, model_index)
        return state_wrapper.get_max_q_value()

    def get_q(self, state: Any, action: int, model_index: int = 0):
        state_wrapper: State = self.__get_state(state, model_index)
        return state_wrapper.optimal_actions[action].q

    def get_v(self, state: Any, model_index: int = 0):
        state_wrapper: State = self.__get_state(state, model_index)
        return state_wrapper.v

    def update_q(self, state: Any, action: int, q: float, episode_done: bool, model_index: int = 0):
        state_wrapper: State = self.__get_state(state, model_index)
        state_wrapper.optimal_actions[action].q = q

    def __get_state(self, state, model_index) -> State:
        return self._models[model_index].states_table[state]

    def update_v(self, state: Any, v: float, episode_done: bool, model_index: int = 0):
        state_wrapper: State = self.__get_state(state, model_index)
        state_wrapper.v = v

    def update_q_values(self, state: Any, values, episode_done: bool, model_index: int = 0):
        raise Exception("Table model did not implement this method")



