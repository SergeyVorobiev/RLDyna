import random
from enum import Enum
from matplotlib import pyplot as plt

cheese_p_black = 0.9
cheese_p_white = 0.1


class Reward(Enum):
    CHEESE = 1
    HIT = -1


def prob(value):
    accuracy = 100
    disp = int(value * accuracy)
    result = random.randrange(0, accuracy, 1)
    if result < disp:
        return 1
    else:
        return 0


def try_box(prob_value, rewards) -> Reward:
    value = prob(prob_value)
    reward = Reward.HIT
    if value == 1:
        reward = Reward.CHEESE
    if rewards is not None:
        rewards.append(reward.value)
    return reward


def try_black_box(rewards=None) -> Reward:
    return try_box(cheese_p_black, rewards)


def try_white_box(rewards=None) -> Reward:
    return try_box(cheese_p_white, rewards)


def q(proba):
    return proba - (1 - proba)

def random_pick(qs):
    size = len(qs)
    return random.randrange(0, size, 1)

def greedy_rand(qs, chance):
    yes = prob(chance)
    if yes:
        return greedy(qs)
    return random_pick(qs)

def greedy(qs):
    maximum = qs[0]
    index = 0
    for i in range(len(qs)):
        val = qs[i]
        if val > maximum:
            maximum = val
            index = i
    return index


def Q(rewards):
    size = len(rewards)
    if size == 0:
        return 0
    return sum(rewards) / size


def Q_exp(rewards, expected_reward):
    size = len(rewards)
    if size == 0:
        return expected_reward
    return (sum(rewards) + expected_reward) / size + 1


def v(Q: []):
    return max(Q)


def regret(v, actual_q):
    # 0.8 - (-0.8) = 1.6
    return 1 - actual_q



if __name__ == '__main__':

    # q(white) = 0.1 - (1 - 0.1) = -0.8
    # q(black) = 0.9 - (1 - 0.9) = 0.8
    v_white = q(cheese_p_white)  # true q q(p) = v, actual q q(a)
    v_black = q(cheese_p_black)  # true q q(p) = v, actual q q(a)
    tries = 1000
    white_rewards = [1]  # 1
    black_rewards = [-1]  # -1
    rewards = []
    actual_white_q_value = 0
    actual_black_q_value = 0
    average_white_regret = 0
    average_black_regret = 0
    white_regrets = []  # -1.8
    black_regrets = []  # 0.2
    regret_values = []
    regret_white_values = []
    regret_black_values = []
    sum_regret = 0
    sum_regret_white = 0
    sum_regret_black = 0
    v_max_value = 0
    for try_number in range(1, tries + 1):
        white_reward = try_white_box()
        black_reward = try_black_box()
        white_q_value = Q(white_rewards)
        black_q_value = Q(black_rewards)
        v_value = v([white_q_value, black_q_value])
        if v_value > v_max_value:
            v_max_value = v_value
        regret_white_value = regret(v_max_value, white_q_value)
        regret_black_value = regret(v_max_value, black_q_value)
        choice = greedy_rand([white_q_value, black_q_value], 0.95)
        if choice == 0:
            white_rewards.append(white_reward.value)
            q_value = white_q_value
        else:
            black_rewards.append(black_reward.value)
            q_value = black_q_value

        regret_value = regret(v_max_value, q_value)
        sum_regret = sum_regret + regret_value
        sum_regret_white = sum_regret_white + regret_white_value
        sum_regret_black = sum_regret_black + regret_black_value
        regret_values.append(sum_regret)
        regret_white_values.append(sum_regret_white)
        regret_black_values.append(sum_regret_black)
        print("White regret: " + str(white_q_value))
        print("Black regret: " + str(black_q_value))
        print("Regret: " + str(regret_value))
    plt.plot(regret_values, label="Regret", color="blue")
    plt.plot(regret_black_values, label="Regret black", color="black")
    plt.plot(regret_white_values, label="Regret white", color="grey")
    plt.legend(loc="upper left")
    plt.show()
