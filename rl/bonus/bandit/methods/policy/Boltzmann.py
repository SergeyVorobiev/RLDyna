import math
import random

from rl.helpers.RMath import RMath


class Boltzmann(object):

    def __init__(self, actions_count, name="Boltzmann"):
        self.name = name
        self.actions_count = actions_count
        self.actions_pick_count = []
        self.__sum_rewards = 0
        self.__rewards_tracker = []
        self.qs: [] = []
        self.prefs = []
        self.policies = []
        self.t = 0
        for _ in range(actions_count):
            self.actions_pick_count.append(0)
            self.qs.append(0.0)
            self.prefs.append(0.0)
            self.policies.append(0.0)

    def add_reward(self, reward, action: int):
        self.t += 1
        self.__sum_rewards = self.__sum_rewards + reward
        self.__rewards_tracker.append(self.__sum_rewards)
        self.actions_pick_count[action] += 1
        qs_prev = []
        for q in self.qs:
            qs_prev.append(q)
        self.qs[action] = RMath.q(self.qs[action], reward, self.actions_pick_count[action])
        self.__calc_prefs(action, qs_prev)
        self.__update_policies()
        # self.__calc_prefs(action, reward)
        # self.__update_policies()

    def __calc_prefs(self, action, qs_prev):
        for a in range(self.actions_count):
            actions_count = self.actions_pick_count[a]
            if actions_count == 0:
                alpha = 0
            else:
                alpha = 1 / actions_count
            if a == action:
                self.prefs[a] = self.prefs[a] + alpha * (self.qs[a] - qs_prev[a]) * (1 - self.policies[a])
            else:
                self.prefs[a] = self.prefs[a] - alpha * (self.qs[a] - qs_prev[a]) * self.policies[a]

    def get_sum_rewards(self):
        return self.__sum_rewards

    def get_rewards_tracker(self):
        return self.__rewards_tracker

    def get_name(self):
        return self.name

    def get_action(self):
        return self.__e_greedy_rand()

    def __e_greedy_rand(self):
        spread = 5
        value = random.randrange(1, spread, 1) / 100
        e = round(1 - value, 2)
        return RMath.greedy_rand(self.policies, e)

    def __update_policies(self):
        e_h = []
        for pref in self.prefs:
            e_h.append(math.pow(math.e, pref))
        sum_e_h = sum(e_h)
        for i in range(self.actions_count):
            self.policies[i] = e_h[i] / sum_e_h
