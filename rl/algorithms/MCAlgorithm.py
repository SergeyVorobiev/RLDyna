from abc import abstractmethod
from typing import Any

from rl.algorithms.RAlgorithm import RAlgorithm
from rl.algorithms.helpers.MCHelper import MCHelper
from rl.models.RModel import RModel


# Monte Carlo
class MCAlgorithm(RAlgorithm):

    # memory_capacity - means that it will keep the n steps in memory (for 10 - 0-9, 1-10, 2-11...)
    # batch_size - means the amount of steps that will be handled by nn-model in one time, 0 means all steps that memory
    # keeps will be handled by nn in one time.
    # clear_memory_every_n_steps - the algorithm sends steps to learn after the episode is done or if the memory is
    # full, but if we do not clear the memory it will be full forever, meaning that the algorithm will send
    # a batch of steps to the nn every step (1-10, 2-11, 3-12, 4-13 for n=10) this flag allows sending steps this way
    # (1-10, 10-20, 20-30)
    # tail_method - def tail(models, next_state) return nextU, specify this method if you want to add a tail
    # when the episode is done by not finish but by timeout
    # terminal_state_checker - when the episode is done, this method allows additionally to check that we achieve the
    # terminal state, because if yes - we do not need to add a tail, but if not we probably want to add a tail.
    def __init__(self, alpha: float = 1, discount: float = 1, memory_capacity: int = 1, batch_size: int = 0,
                 clear_memory_every_n_steps=False, tail_method=None, terminal_state_checker=None):
        super().__init__(alpha=alpha, discount=discount, memory_capacity=memory_capacity,
                         terminal_state_checker=terminal_state_checker)
        self._batch_size = batch_size
        self._clear_memory_every_n_steps = clear_memory_every_n_steps
        self._tail_method = tail_method
        self._n_learn_steps = 0
        self._steps_count = 0
        self._last_discount = 1.0

    # It allows learning not when the memory is full but after specified amount of steps, for memory_capacity = 10 and
    # n_steps = 5 it will send the steps to nn as - (0-5, 0-10, 5-15, 10-20...)
    # useless for MC, use it only for TD
    def learn_every_n_steps(self, n_steps):
        self._n_learn_steps = n_steps

    def train(self, models: [RModel]) -> (float, Any):
        batch = self.get_last_memorized()
        _, _, _, n_state, done, truncated, props = batch[0]
        self._steps_count += 1

        used_tail = False
        plan_batch = None

        # So as this is a Monte Carlo we are waiting for the end of the episode
        # But we can use it as TD(N)
        if done or self._steps_count == self._n_learn_steps or self.is_memory_full():

            # Means that we do not want to learn because memory is full, but each n steps if the n steps is 0
            # then we will learn when memory is full
            # When done we always learn
            if not done and self._n_learn_steps > 0 and self._steps_count != self._n_learn_steps:
                return 0, None

            self._steps_count = 0
            terminal = done and self._terminal_state_checker(truncated, props)
            start_g = 0

            # If not done we consider that the algorithm is TD(N) and we need to add tail
            if not terminal and self._tail_method is not None:
                used_tail = True
                start_g = self._tail_method(models, n_state)

            # For each sample of the episode we accumulate reward (G) and form batch to feed it to NN
            # In this case we use from end to start approach
            # Technically we could start from start, but then we have to get the sum of rewards for every step:
            # 1 - 5, 2 - 5, 3 - 5, 4 - 5, 5 - 5, but starting from end we just get 4, 3, 2, 1, 0
            # and calculate discount more easily

            batch = self.get_last_memorized(self.get_memory_size())
            gs, self._last_discount = MCHelper.build_g(batch, self._discount, start_g=start_g, need_reverse=True,
                                                       used_tail=used_tail, last_used_discount=self._last_discount)
            if done:
                self._last_discount = 1.0

            i = 0

            # Reverse from start to end because we can use this as TD(N)
            batch.reverse()
            plan_batch = []
            for state, action, reward, next_state, done, truncated, props in batch:
                plan_batch.append([state, action, float(reward), next_state, done, truncated, float(gs[i][0]),
                                   float(gs[i][1]), props])
                self.pick_data(state, action, float(reward), next_state, done, truncated, float(gs[i][0]),
                               float(gs[i][1]), props)
                i += 1

            self.send_data_to_model(models)
            if self._clear_memory_every_n_steps:
                self.clear_memory()
        return 0, plan_batch

    @abstractmethod
    def pick_data(self, state, action, reward, next_state, done, truncated, state_g, state_discount, props):
        ...

    @abstractmethod
    def send_data_to_model(self, models):
        ...

    def get_action_values(self, models: [RModel], state: Any):
        return models[0].get_action_values(state)

    def plan(self, models: [RModel], batch) -> (float, Any):
        for state, action, reward, next_state, done, truncated, state_g, state_discount, props in batch:
            self.pick_data(state, action, reward, next_state, done, truncated, state_g, state_discount, props)
        self.send_data_to_model(models)

    def get_v(self, models: [RModel], state: Any) -> float:
        pass

    def get_q_values(self, models: [RModel], state: Any):
        pass

    def update_policy(self):
        pass
