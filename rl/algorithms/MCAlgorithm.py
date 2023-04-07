from abc import abstractmethod
from typing import Any

from rl.algorithms.RAlgorithm import RAlgorithm
from rl.algorithms.helpers.MCHelper import MCHelper
from rl.models.RModel import RModel


# Monte Carlo
class MCAlgorithm(RAlgorithm):

    def __init__(self, alpha: float = 1, discount: float = 1, memory_capacity: int = 1):
        super().__init__(alpha=alpha, discount=discount, memory_capacity=memory_capacity)

    def train(self, models: [RModel]) -> (float, Any):
        batch = self.get_last_memorized()
        _, _, _, _, done, _ = batch[0]

        # So as this is a Monte Carlo we are waiting for the end of the episode
        if done:

            # For each sample of the episode we accumulate reward (G) and form batch to feed it to NN
            # In this case we use from end to start approach
            # Technically we could start from start, but then we have to get the sum of rewards for every step:
            # 1 - 5, 2 - 5, 3 - 5, 4 - 5, 5 - 5, but starting from end we just get 4, 3, 2, 1, 0
            # and calculate discount more easily

            batch = self.get_last_memorized(self.get_memory_size())
            gs = MCHelper.build_g(batch, self._discount)
            i = 0
            for state, action, reward, next_state, done, props in batch:
                self.pick_data(state, action, reward, next_state, done, gs[i][0], gs[i][1], props)
                i += 1

            self.send_data_to_model(models)
        return 0, batch

    @abstractmethod
    def pick_data(self, state, action, reward, next_state, done, state_g, state_discount, props):
        ...

    @abstractmethod
    def send_data_to_model(self, models):
        ...

    def get_a_distribution(self, models: [RModel], state: Any):
        return models[0].get_a_distribution(state)

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
