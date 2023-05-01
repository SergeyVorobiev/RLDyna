from rl.mdp.Action import Action
from rl.mdp.State import State
from rl.mdp.Transition import Transition


class TransitionMap(Transition):

    def __init__(self, transition_map):
        self.transition_map = transition_map

    def get_transition(self, s: State, a: Action, sn: State) -> (float, float):
        s_id = s.state_id
        sn_id = sn.state_id
        try:
            probability = self.transition_map[s_id][a.key][sn_id][0]
        except KeyError:
            return 0, 0
        return probability, self.transition_map[s_id][a.key][sn_id][1] * probability


class TransitionMapBuilder(object):

    def __init__(self):
        self.transition_map = {}

    def add_transition(self, state_id: int, action_name: str, next_state_id: int, probability: float, reward=None):
        state_map = self.transition_map.get(state_id)
        if state_map is None:
            state_map = {}
            probability_map = {next_state_id: [probability, reward]}
            state_map[action_name] = probability_map
            self.transition_map[state_id] = state_map
        elif state_map.get(action_name) is None:
            probability_map = {next_state_id: [probability, reward]}
            state_map[action_name] = probability_map
        else:
            state_map[action_name][next_state_id] = [probability, reward]
        return self

    def build(self):
        return TransitionMap(self.transition_map)
