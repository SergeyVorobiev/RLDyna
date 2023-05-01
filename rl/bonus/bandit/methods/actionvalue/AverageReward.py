from rl.helpers.RMath import RMath


class AverageReward(object):

    def __init__(self, actions_count, algorithm):
        self.t = 0
        self.algorithm = algorithm
        self.actions_count = actions_count
        self.actions_pick_count = []
        self.__sum_rewards = 0
        self.__rewards_tracker = []
        self.qs: [float] = []
        for _ in range(actions_count):
            self.actions_pick_count.append(0)
            self.qs.append(0.0)

    def add_reward(self, reward, action: int):
        self.t += 1
        self.__sum_rewards = self.__sum_rewards + reward
        self.__rewards_tracker.append(self.__sum_rewards)
        self.actions_pick_count[action] += 1
        self.qs[action] = RMath.q(self.qs[action], reward, self.actions_pick_count[action])

    def get_action(self):
        return self.algorithm(self.qs, self.actions_pick_count, self.t)

    def get_sum_rewards(self):
        return self.__sum_rewards

    def get_rewards_tracker(self):
        return self.__rewards_tracker

    def get_name(self):
        return self.algorithm.__name__
