from rl.models.rewardestimators.RewardEstimator import RewardEstimator


class MountainCarRewardEstimator(RewardEstimator):

    def __init__(self):
        self._iter = 0
        self._max_iter = 200

    # Not mandatory, just for testing
    def estimate(self, state, action, reward, next_state, done, env_props) -> float:
        self._iter += 1
        if done:
            ratio = self._max_iter / self._iter
            reward = reward + next_state[0] * ratio
            self._iter = 0
        return reward
