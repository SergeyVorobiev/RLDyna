from typing import Any


class MCHelper:

    def __init__(self):
        pass

    # return discounted G & discounts array
    # take memory_batch from end to start
    # memory_batch should have from end to start order
    @staticmethod
    def build_g(memory_batch: [Any], discount: float, need_reverse: bool = False) -> [[float, float]]:
        acc_discount = 1.0
        size = memory_batch.__len__()
        discounts = []
        result = []

        # compute discount for 0.9 we should get 1, 0.9, 0.9 * 0.9, 0.9 * 0.9 * 0.9...
        for i in range(size):
            discounts.append(acc_discount)
            acc_discount *= discount

        # reverse because we count memory_batch from end to start but discounts now from start to end
        discounts.reverse()
        g = 0
        i = 0
        for _, _, reward, _, _, _ in memory_batch:
            gamma = discounts[i]
            g = reward + gamma * g
            result.append([g, gamma])

        # if need to get array from start to end order
        if need_reverse:
            result.reverse()
        return result
