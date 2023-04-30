from typing import Any


# This helper is intended to calculate g and discount for Monte Carlo, but as Monte Carlo is an extreme case of any
# temporal difference algorithm such as (SARSA, Q, TD etc.) we can also use this to calculate N step temporal difference
# specify start_g as a tail so as we build_acd g from end to start, the start_g is actually an end.
class MCHelper:

    def __init__(self):
        pass

    # return discounted G & discounts array
    # take memory_batch from end to start
    # memory_batch should have 'from end to start' order
    # need_reverse - if true then it converts result to be in 'from start to end' order
    # start_g is a tail U' or Q' in case if we use some form of TD(N) and the episode is not done we need to add a tail
    @staticmethod
    def build_g(memory_batch: [Any], discount: float, start_g: float = 0.0,
                need_reverse: bool = False, used_tail: bool = False) -> [[float, float]]:
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

        # if start_g == 0 then discount here does not important, but if we use TD then we need to apply the last one
        # to the tail
        g = start_g * acc_discount  # it should be the same as for the last reward

        # because tail can theoretically be 0 we need to check the tail by a flag
        if used_tail:
            result.append([g, acc_discount])
        i = 0
        for _, _, reward, _, _, _, _ in memory_batch:
            if reward == -100:
                reward = -1
            gamma = discounts[i]
            g += reward * gamma
            result.append([g, gamma])
            i += 1

        # if it needs to get array from start to end order
        if need_reverse:
            result.reverse()
        return result
