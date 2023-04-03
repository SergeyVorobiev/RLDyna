from rl.rewardestimators.RewardEstimator import RewardEstimator


class CartPoleRewardEstimator(RewardEstimator):

    def estimate(self, state, action, reward, next_state, done, env_props) -> float:
        return reward
