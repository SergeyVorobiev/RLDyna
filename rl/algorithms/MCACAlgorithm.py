from abc import abstractmethod
from enum import Enum
from typing import Any

import tensorflow
from rl.algorithms.MCAlgorithm import MCAlgorithm
from rl.models.RModel import RModel


class UseCritic(Enum):
    No = 0
    Yes = 1

    # It will not be used to calculate sigma, but it can calculate U-tail for TD
    NoButLearn = 2


# Monte Carlo Policy-Gradient Continuous + Critic
class MCACAlgorithm(MCAlgorithm):

    def __init__(self, discount: float = 1, memory_capacity: int = 1, batch_size=0,
                 clear_memory_every_n_steps=False, tail_method=None, terminal_state_checker=None):
        super().__init__(discount=discount, memory_capacity=memory_capacity, batch_size=batch_size,
                         clear_memory_every_n_steps=clear_memory_every_n_steps, tail_method=tail_method,
                         terminal_state_checker=terminal_state_checker)
        tensorflow.get_logger().setLevel("ERROR")
        self._baseline_from_train_step = False
        self._use_critic = UseCritic.Yes
        self._shuffle_critic = False
        self._shuffle_actor = False
        self._x = []
        self._y = []
        self._y2 = []

    def get_action_values(self, models: [RModel], state: Any):
        return models[0].get_action_values(state)

    @abstractmethod
    def pick_data(self, state, action, reward, next_state, done, truncated, state_g, state_discount, props):
        ...

    def use_critic(self, use: UseCritic):
        self._use_critic = use

    def use_baseline_from_train_step(self, baseline_from_train_step):
        self._baseline_from_train_step = baseline_from_train_step

    def shuffle_actor_steps(self, shuffle: bool):
        self._shuffle_actor = shuffle

    def shuffle_critic_steps(self, shuffle: bool):
        self._shuffle_critic = shuffle

    def send_data_to_model(self, models):

        # Specify batch size for actor
        if self._batch_size > 0:
            batch_s = self._batch_size
        else:
            batch_s = self._x.__len__()

        # Train Critic
        losses = None
        if self._use_critic == UseCritic.Yes or self._use_critic == UseCritic.NoButLearn:
            if self._baseline_from_train_step:
                history, losses = models[1].update((self._x, self._y2), 1)
            else:
                models[1].update((self._x, self._y2), self._x.__len__(), shuffle=self._shuffle_critic)
                pred = models[1].predict(self._x)
                losses = []
                for i in range(self._y2.__len__()):
                    losses.append(self._y[i][0] - float(pred[i]))

        # Replace losses by critic
        if self._use_critic == UseCritic.Yes:
            for i in range(self._y.__len__()):
                self._y[i][0] = losses[i]

        models[0].update((self._x, self._y), batch_s, use_loss_callback=False, shuffle=self._shuffle_actor)
        self._x.clear()
        self._y.clear()
        self._y2.clear()

    @abstractmethod
    def pick_action(self, models: [RModel], state: Any) -> Any:
        ...
