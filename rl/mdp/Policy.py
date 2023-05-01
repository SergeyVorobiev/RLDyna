from abc import abstractmethod

from rl.mdp.Action import Action
from rl.mdp.State import State


class Policy:

    def __init__(self):
        pass

    @staticmethod
    def build_uniform_policy(states: [State]):
        if states is None or states.__len__() == 0:
            raise Exception("Can not build uniform policy without states.")
        state: State
        for state in states:
            action: Action
            k = 0
            for action in state.get_actions():
                state.optimal_actions[k] = action
                action.priority = k
                k += 1
            state.optimal_actions_count = state.get_actions().__len__()
            state.probability = 1 / state.optimal_actions_count

    @staticmethod
    def build_constant_policy(action_key, states: [State]):
        if states is None or states.__len__() == 0:
            raise Exception("Can not build uniform policy without states.")
        state: State
        for state in states:
            state.probability = 0
            actions = state.get_actions()
            size = actions.__len__()
            for action in state.get_actions():
                if action.key == action_key:
                    state.optimal_actions_count = 1
                    state.optimal_actions[0] = action  # optimal action go to the start
                    state.probability = 1
                else:
                    size = size - 1
                    state.optimal_actions[size] = action  # not optimal actions go to the end

    @staticmethod
    def make_actions_optimal_by_q_in_state(state: State) -> int:
        policy_changed = 0
        action: Action
        actions = state.optimal_actions
        max_q = actions[0].q
        optimal_action_count = 1
        size = actions.__len__()
        end = size - 1
        start = 0
        while start != end:
            action = actions[end]
            q = action.q
            if q > max_q:
                policy_changed += 1
                # for count in range(optimal_action_count):  # Not optimal actions goes to the end
                actions[end] = actions[0]
                max_q = q
                actions[0] = action
                optimal_action_count = 1
                end -= 1
            elif q == max_q:
                # if state.optimal_actions[optimal_action_count].a_id != action.a_id:
                if action.q != actions[optimal_action_count].q:
                    policy_changed += 1
                    actions[end] = actions[optimal_action_count]
                    actions[optimal_action_count] = action
                optimal_action_count += 1
                start += 1
            else:
                end -= 1
        state.max_q = max_q
        state.optimal_actions_count = optimal_action_count
        state.probability = 1 / state.optimal_actions_count
        return policy_changed

    @staticmethod
    def make_actions_optimal_by_q(states: [State]) -> int:
        policy_changed = 0
        state: State
        for state in states:
            policy_changed += Policy.make_actions_optimal_by_q_in_state(state)
        return policy_changed

    @abstractmethod
    def pick_action(self, state: State) -> Action:
        ...

    @abstractmethod
    def get_optimal_action(self, state: State) -> Action:
        ...

    @abstractmethod
    def pick_actions(self, state: State) -> {object: Action}:
        ...
