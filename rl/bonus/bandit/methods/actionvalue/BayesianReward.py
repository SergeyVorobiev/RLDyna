from scipy.stats import beta

from rl.helpers.RMath import RMath


class BayesianReward(object):

    def __init__(self, actions_count, c=1, name="Bayesian"):
        self.t = 0
        self.name = name
        self.c = c
        self.actions_count = actions_count
        self.actions_pick_count = []
        self.__sum_rewards = 0
        self.__a_b_rewards = []
        self.__rewards_tracker = []
        self.qs: [float] = []
        self.beta_std = []
        for _ in range(actions_count):
            self.actions_pick_count.append(0)
            self.qs.append(0.0)
            self.__a_b_rewards.append([0, 0])
            self.beta_std.append(beta(1, 1).std() * self.c)

    def add_reward(self, reward, action: int):
        self.t += 1
        self.__sum_rewards = self.__sum_rewards + reward
        self.__update_a_b(reward, action)
        self.__rewards_tracker.append(self.__sum_rewards)
        self.actions_pick_count[action] += 1
        self.qs[action] = RMath.q(self.qs[action], reward, self.actions_pick_count[action])
        self._update_beta_std(action)

    def __update_a_b(self, reward, action):
        a_b = self.__a_b_rewards[action]
        if reward < 0:
            a_b[0] = a_b[0] + abs(reward)
        elif reward > 0:
            a_b[1] = a_b[1] + reward

    def get_action(self):
        return RMath.greedy(self.beta_std)

    def get_sum_rewards(self):
        return self.__sum_rewards

    def get_rewards_tracker(self):
        return self.__rewards_tracker

    def get_name(self):
        return self.name

    def _update_beta_std(self, action):
        a_b = self.__a_b_rewards[action]
        a = a_b[0]
        b = a_b[1]
        if a == 0:
            a = 1
        if b == 0:
            b = 1
        self.beta_std[action] = beta(a, b).std() * self.c + self.qs[action]
