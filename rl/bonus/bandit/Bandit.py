from random import shuffle

from scipy.stats import norm
from matplotlib import pyplot as plt


class Bandit(object):

    def __init__(self, mean, std, cached_results=1000, allow_regenerate_cache=True):
        self.__dist = norm(mean, std)
        self.__pointer = 0
        self.__allow_regenerate_cache = allow_regenerate_cache
        self.__cached_results = cached_results + 1
        self.values = self.__generate()

    def __generate(self):
        values = self.__dist.rvs(size=self.__cached_results)

        # convert to int rewards
        for i in range(self.__cached_results):
            values[i] = round(values[i], 0)
        values = values.astype(int)

        # just because I want
        shuffle(values)
        return values

    def get_cached_results(self):
        return self.__cached_results

    def reset(self):
        self.__pointer = 0

    def pull_hand(self) -> int:
        value = self.values[self.__pointer]
        self.__pointer = self.__pointer + 1
        if self.__pointer == self.__cached_results:
            if self.__allow_regenerate_cache:
                self.values = self.__generate()
                self.__pointer = 0
            else:
                raise Exception("Cache end")
        return value

    def plot(self):
        plt.hist(self.values, density=True, histtype='stepfilled', alpha=0.9)
        plt.show()
