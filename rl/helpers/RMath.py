import math
import random

import statistics
from scipy.stats import beta
from sklearn.preprocessing import normalize


class RMath(object):

    @staticmethod
    def normalization(data: [], norm="l2"):
        normalize([data], norm=norm)  # [1, 2, 3, 4] array([[0.18257419, 0.36514837, 0.54772256, 0.73029674]])
        # [1, 2, 3, 4] array([[0.1, 0.2, 0.3, 0.4]]) "l1"
        # [1, 2, 3, 4] array([[0.25, 0.5 , 0.75, 1.]]) "max"

    @staticmethod
    def q(q_old, reward, n):
        return q_old + 1 / n * (reward - q_old)

    @staticmethod
    def min(qs):
        min_value = qs[0]
        index = 0
        for i in range(1, len(qs)):
            q = qs[i]
            if min_value > q:
                min_value = q
                index = i
        return index

    @staticmethod
    def qs(pick, qs, b_rewards):
        rewards = b_rewards[pick]
        size = len(rewards)
        qs[pick] = RMath.q(qs[pick], rewards[size - 1], size)
        return qs

    @staticmethod
    def qs2(pick, qs, q1, b_rewards):
        rewards = b_rewards[pick]
        qs[pick] = RMath.q_non_stationary(q1, rewards)
        return qs

    @staticmethod
    def q_non_stationary(q1, rewards):
        n = len(rewards)
        alpha = 1 / n
        one_alpha = 1 - alpha
        head = q1 * pow(one_alpha, n)
        tail = 0
        for i in range(1, n + 1):
            reward = rewards[i - 1]
            tail = tail + alpha * pow(one_alpha, n - i) * reward
        return head + tail

    @staticmethod
    def q_exp(rewards, expected_reward):
        size = len(rewards)
        if size == 0:
            return expected_reward
        return (sum(rewards) + expected_reward) / size + 1

    @staticmethod
    def v(q: []):
        return max(q)

    @staticmethod
    def regret(v, actual_q):
        # 0.8 - (-0.8) = 1.6
        return v - actual_q

    @staticmethod
    def random_pick(values):
        size = len(values)
        return random.randrange(0, size, 1)

    @staticmethod
    def greedy_rand(values, chance):
        yes = RMath.get_value(chance)
        if yes:
            return RMath.greedy(values)
        return RMath.random_pick(values)

    @staticmethod
    def greedy(values):
        maximum = values[0]
        index = 0
        for i in range(1, len(values)):
            val = values[i]
            if val > maximum:
                maximum = val
                index = i
        return index

    @staticmethod
    def usb(qs, sizes, t, param_value):
        index = 0
        maximum = 0
        first = True
        for i in range(len(qs)):
            q_value = qs[i]
            size = sizes[i]
            q_value = q_value + RMath.__u(size, t, param_value)
            if first:
                maximum = q_value
                index = i
                first = False
            elif q_value > maximum:
                maximum = q_value
                index = i
        return index

    @staticmethod
    def __u(size, t, param_value):
        if size == 0:
            return 0
        return math.sqrt(math.log(t) / size) * param_value

    @staticmethod
    def bayesian(qs, a_b_arr, c=1):
        result = []
        for i in range(len(qs)):
            q = qs[i]
            a_b = a_b_arr[i]
            a = a_b[0]
            b = a_b[1]
            if a < 1:
                a = 1
            if b < 1:
                b = 1
            result.append(q + c * beta(a, b).std())
        return RMath.greedy(result)

    @staticmethod
    def bayesian2(qs, all_rewards, c=1):
        result = []
        for i in range(len(qs)):
            q = qs[i]
            rewards = all_rewards[i]
            size = len(rewards)
            if size < 2:
                alpha = 0
            else:
                alpha = c * statistics.stdev(rewards)
            result.append(q + alpha)
        return RMath.greedy(result)

    @staticmethod
    def get_value(probability, value1=1, value2=0):
        accuracy = 100
        dispersion = int(probability * accuracy)
        result = random.randrange(0, accuracy, 1)
        if result < dispersion:
            return value1
        else:
            return value2

    @staticmethod
    def get_values(probability, count, value1=1, value2=0):
        result = []
        for i in range(count):
            result.append(RMath.get_value(probability, value1, value2))
        return result

    @staticmethod
    def __is_arrays_equal(array1: [], array2: []):
        if array1 is None and array2 is None:
            return True
        elif (array1 is None and array2 is not None) or (array1 is not None and array2 is None):
            return False
        size1 = len(array1)
        size2 = len(array2)
        if size1 == 0 and size2 == 0:
            return False
        if size1 != size2:
            return False
        for v1, v2 in zip(array1, array2):
            if v1 != v2:
                return False
        return True
