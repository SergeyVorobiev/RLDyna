from typing import Union

from rl.mdp.Action import Action
from rl.mdp.Policy import Policy
from rl.mdp.State import State
from rl.mdp.Transition import Transition
from rl.mdp.TransitionMap import TransitionMapBuilder


class MDP(object):

    def __init__(self, discount=1):
        self.__reward_f = None
        self.__states: [State] = []
        self.__transitions: Transition or None = None
        self.__actions = None
        self.__discount = discount
        self.__each_expected_calc = self.__each_expected_calc_f
        self.__sweep_expected_calc = self.__sweep_expected_calc_f
        self.__evaluation_calc_callback = self.__evaluation_v_f
        self.__improve_calc_callback = self.__sweep_expected_calc_f
        self.__improve_end_callback = self.__sweep_expected_calc_f

    @staticmethod
    def transition_map_builder() -> TransitionMapBuilder:
        return TransitionMapBuilder()

    def __each_expected_calc_f(self, state: State):
        pass

    def __sweep_expected_calc_f(self, states: [State]):
        pass

    def __evaluation_v_f(self, states: [State], iterations):
        pass

    def get_actions(self) -> {object: Action}:
        return self.__actions

    def set_transitions(self, transitions: Transition):
        self.__transitions = transitions

    def get_expected_values(self):
        result = []
        for state in self.__states:
            result.append([state.name, state.v])
        return result

    def add_state(self, state_name, actions: {object: Union[Action, str, int]} or [Union[Action, str, int]], info=None,
                  visual=None):
        state_id = self.__states.__len__()
        state = State(state_id, state_name, actions, info=info, visual=visual)
        self.__states.append(state)
        return state_id

    def get_states(self):
        return self.__states

    def evaluate_state_values(self, epsilon, stop_iteration_count=1000):
        iterations = 0
        error = epsilon + 1
        while error > epsilon and iterations < stop_iteration_count:
            iterations += 1
            error = self.__calc_v2()
            self.__sweep_expected_calc(self.__states)
        self.__evaluation_calc_callback(self.__states, iterations)
        return iterations

    def build_uniform_policy(self):
        Policy.build_uniform_policy(self.__states)

    def build_constant_policy(self, action_key):
        Policy.build_constant_policy(action_key, self.__states)

    def register_callback_for_each_expected_value_calc(self, each_expected_calc):
        self.__each_expected_calc = each_expected_calc

    def register_callback_for_sweep_expected_values_calc(self, sweep_expected_calc):
        self.__sweep_expected_calc = sweep_expected_calc

    def register_callback_for_evaluation(self, evaluation_callback):
        self.__evaluation_calc_callback = evaluation_callback

    def register_callback_for_improvement(self, improvement_callback):
        self.__improve_calc_callback = improvement_callback

    def register_improve_end_callback(self, improvement_end_callback):
        self.__improve_end_callback = improvement_end_callback

    def __calc_q(self, state, action):
        q = 0
        for next_state in self.__states:
            next_v = next_state.v
            transition_probability, expected_transition_reward = self.__transitions.get_transition(state, action,
                                                                                                   next_state)
            q += (transition_probability * next_v + expected_transition_reward)
            if transition_probability == 1:
                break
        q *= self.__discount
        action.q = q
        return q

    @staticmethod
    def __calc_error(old_value, new_value, max_error):
        error = abs(old_value - new_value)
        if max_error is None:
            max_error = error
        elif max_error < error:
            max_error = error
        return max_error

    def __calc_v(self) -> float:
        max_error = None
        state: State
        for state in self.__states:
            v_value = 0
            for i in range(0, state.optimal_actions_count):
                action = state.optimal_actions[i]
                v_value += state.probability * self.__calc_q(state, action)
            max_error = MDP.__calc_error(state.v, v_value, max_error)
            state.v = v_value
            state.update_visual()
            self.__each_expected_calc(state)
        return max_error

    def __calc_v2(self) -> float:
        max_error = None
        state: State
        for state in self.__states:
            max_value = None
            for i in range(0, state.action_size):
                action = state.optimal_actions[i]
                self.__calc_q(state, action)
                if max_value is None or action.q > max_value:
                    max_value = action.q
            max_error = MDP.__calc_error(state.v, max_value, max_error)
            state.v = max_value
            state.update_visual()
            self.__each_expected_calc(state)
        return max_error

    def __improve_policy(self) -> int:
        for state in self.__states:
            for i in range(state.optimal_actions_count, state.action_size):
                action = state.optimal_actions[i]
                self.__calc_q(state, action)
        return Policy.make_actions_optimal_by_q(self.__states)

    def gpi(self, evaluation_epsilon, evaluation_iteration_stop_count=100, improve_iteration_stop_count=100):
        improve_stop_count = 0
        changed_actions = 1
        min_evaluation_epsilon = 0.1
        last_count = None
        while changed_actions > 0 and improve_stop_count <= improve_iteration_stop_count:
            improve_stop_count += 1
            # print("Evaluation")
            _ = self.evaluate_state_values(epsilon=evaluation_epsilon,
                                           stop_iteration_count=evaluation_iteration_stop_count)
            #if evaluation_epsilon > min_evaluation_epsilon:
            #    print("Evaluation end with count of iterations: " + str(iterations))
            #    print("Improve")
            changed_actions = self.__improve_policy()
            if last_count is not None and last_count < changed_actions:
                evaluation_epsilon *= 0.9
                if evaluation_epsilon < min_evaluation_epsilon:
                    min_evaluation_epsilon = 0.9
                print("Eval eps: " + str(evaluation_epsilon))
            last_count = changed_actions
            print(changed_actions)
            self.__improve_calc_callback(self.__states)
            # print("Improve end with count of changed actions: " + str(changed_actions))
        self.__improve_end_callback(self.__states)


if __name__ == '__main__':
    mdp = MDP()
    face_id = mdp.add_state("facebook", ["stay", "quit"])
    study1_id = mdp.add_state("study1", ["facebook", "study"])
    study2_id = mdp.add_state("study2", ["sleep", "study"])
    study3_id = mdp.add_state("study3", ["pub", "sleep"])
    termination_id = mdp.add_state("termination", ["stay"])
    mdp.build_uniform_policy()
    transition_builder = MDP.transition_map_builder() \
        .add_transition(face_id, "stay", face_id, 1, -1) \
        .add_transition(face_id, "quit", study1_id, 1, 0) \
        .add_transition(study1_id, "facebook", face_id, 1, -1) \
        .add_transition(study1_id, "study", study2_id, 1, -2) \
        .add_transition(study2_id, "sleep", termination_id, 1, 0) \
        .add_transition(study2_id, "study", study3_id, 1, -2) \
        .add_transition(study3_id, "pub", study1_id, 0.2, 1) \
        .add_transition(study3_id, "pub", study2_id, 0.4, 1) \
        .add_transition(study3_id, "pub", study3_id, 0.4, 1) \
        .add_transition(study3_id, "sleep", termination_id, 1, 10)

    mdp.set_transitions(transition_builder.build())
    mdp.gpi(evaluation_epsilon=0.001)
    # print(iterations1)
    print(mdp.get_expected_values())
