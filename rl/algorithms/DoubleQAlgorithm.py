import random

from typing import Any
from rl.algorithms.StepControl import StepControl
from rl.models.RModel import RModel


# off policy TD control
class DoubleQAlgorithm(StepControl):

    def update_policy(self):
        pass

    def get_v(self, models: [RModel], state: Any) -> float:
        pass

    def get_q_values(self, models: [RModel], state: Any):
        model1 = models[0]
        model2 = models[1]
        q_values1 = model1.get_q_values(state)
        q_values2 = model2.get_q_values(state)
        result = []
        for i in range(q_values1.__len__()):
            q1 = q_values1[i]
            q2 = q_values2[i]
            result.append((q1 + q2) / 2)
        return result

    def train_sample(self, models: [RModel], state: Any, action: int, reward: float, next_state: Any,
                     done: bool, env_props: Any) -> float:
        model1 = models[0]
        model2 = models[1]
        if random.randrange(0, 2) == 0:
            q1 = model1.get_q(state, action)
            if not done:
                max_a = model1.get_max_a(next_state)
                next_q2 = model2.get_q(next_state, max_a)
                reward = reward + self._discount * next_q2
            q1 = q1 + self._alpha * (reward - q1)
            model1.update_q(state, action, q1, done)
        else:
            q2 = model2.get_q(state, action)
            if not done:
                max_a = model2.get_max_a(next_state)
                next_q1 = model1.get_q(next_state, max_a)
                reward = reward + self._discount * next_q1
            q2 = q2 + self._alpha * (reward - q2)
            model2.update_q(state, action, q2, done)
        return 0
