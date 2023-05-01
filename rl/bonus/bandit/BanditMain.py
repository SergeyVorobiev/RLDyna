import math
import random


from matplotlib import pyplot as plt

from rl.bonus.bandit.Bandit import Bandit
from rl.bonus.bandit.methods.actionvalue.AverageReward import AverageReward
from rl.bonus.bandit.methods.actionvalue.BayesianReward import BayesianReward
from rl.bonus.bandit.methods.policy.Boltzmann import Boltzmann
from rl.helpers.RMath import RMath


class BanditGame(object):

    def __init__(self, bandit_props: [], agents, tries):
        self.tries = tries
        self.bandits: [Bandit] = []
        for props in bandit_props:
            mean = props[0]
            std = props[1]
            self.bandits.append(Bandit(mean=mean, std=std, cached_results=tries, allow_regenerate_cache=False))
        self.agents = agents

    def pick_bandit(self, index):
        bandit: Bandit = self.bandits[index]
        return bandit.pull_hand()

    def reset_bandits(self):
        for bandit in self.bandits:
            bandit.reset()

    def play(self):
        action = 0
        for agent in self.agents:
            self.reset_bandits()
            for i in range(self.tries):
                agent.add_reward(self.pick_bandit(action), action)
                action = agent.get_action()


def greedy1(qs, actions_count, t):
    return RMath.greedy(qs)


def min1(qs, actions_count, t):
    return RMath.min(qs)


def random1(qs, actions_count, t):
    return RMath.random_pick(qs)


def usb1(qs, actions_count, t):
    return RMath.usb(qs, actions_count, t, 1 / 2.7)


def usb2(qs, actions_count, t):
    r = random.randrange(0, 3, 1)
    if r == 0:
        param = 1 / (1.47 * 2)
    elif r == 1:
        param = 1.47
    else:  # r == 2
        param = 1.47 * 2
    return RMath.usb(qs, actions_count, t, param)


def e_greedy(qs, actions_count, t):
    return RMath.greedy_rand(qs, 0.95)


def e_greedy2(qs, actions_count, t):
    return RMath.greedy_rand(qs, 0.97)


def e_greedy_rand(qs, actions_count, t):
    spread = 5
    value = random.randrange(1, spread, 1) / 100
    e = round(1 - value, 2)
    return RMath.greedy_rand(qs, e)


def bayesian2(qs, b_rewards, t):
    return RMath.bayesian2(qs, b_rewards, 2)


def generate_bandit_props(count):
    result = []
    std = 1
    mean = -1
    for i in range(count):
        s = 3 - 1
        addi = s / count
        result.append([mean, std])
        std = std + addi
        mean = mean + addi
    return result


def run():
    print("Calculating...\n")
    bandits_count = 5
    av_agents = []
    algos = [random1, e_greedy_rand, e_greedy2, usb1, usb2]  # min1, random1
    for algo in algos:
        av_agents.append(AverageReward(bandits_count, algo))
    av_agents.append(Boltzmann(bandits_count))
    av_agents.append(BayesianReward(bandits_count))
    game = BanditGame(generate_bandit_props(bandits_count), av_agents, 5000)
    game.play()
    for ag in av_agents:
        plt.plot(ag.get_rewards_tracker(), label=ag.get_name())
        print(ag.get_name() + ": " + str(ag.actions_pick_count))
    print("\nDone")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == '__main__':
    run()

