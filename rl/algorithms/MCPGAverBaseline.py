from typing import Any

from rl.algorithms.RAlgorithm import RAlgorithm
from rl.models.RModel import RModel


# Monte Carlo Policy-Gradient control
# This algorithm shows how we can train agent avoiding using of p and q values
class MCPGAverBaseline(RAlgorithm):

    def __init__(self, alpha: float = 1, discount: float = 1, memory_capacity: int = 1, use_baseline=False):
        super().__init__(alpha=alpha, discount=discount, memory_capacity=memory_capacity)
        self._use_baseline = use_baseline

    def train(self, models: [RModel]) -> (float, Any):
        batch = self.get_last_memorized()
        _, _, _, _, done, _ = batch[0]

        # So as this is a Monte Carlo we are waiting for the end of the episode
        if done:
            model: RModel = models[0]
            g = 0
            states = []
            actions = []
            rewards = []
            baselines = []

            # For each sample of the episode we accumulate reward (G) and form batch to feed it to NN
            # In this case we use from end to start approach
            k = 1
            for state, action, reward, _, _, _ in self.get_last_memorized(self.get_memory_size()):
                g = reward + self._discount * g
                states.append([state])  # pack into a single batch to fid nn properly
                actions.append(action)
                rewards.append(float(g))  # in case if reward int or float64
                baselines.append(self.calculate_baseline(state, g, k))
                k += 1
            model.update([states, actions, rewards, baselines])
        return 0, batch

    def get_a_distribution(self, models: [RModel], state: Any):
        return models[0].get_a_distribution(state)

    def calculate_baseline(self, state, reward=0, state_index=1) -> float:
        if self._use_baseline and state_index > 1:
            return reward / state_index
        return 0.0

    def plan(self, models: [RModel], batch) -> (float, Any):
        pass

    def pick_action(self, models: [RModel], state: Any) -> Any:
        return models[0].get_max_a(state)

    def get_v(self, models: [RModel], state: Any) -> float:
        pass

    def get_q_values(self, models: [RModel], state: Any):
        pass

    def update_policy(self):
        pass
