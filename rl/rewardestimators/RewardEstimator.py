from abc import abstractmethod


# It allows you to build_acd complex reward system instead of default reward
# In Dyna it will be invoked before applying the reward
class RewardEstimator:

    # Returns new reward
    @abstractmethod
    def estimate(self, state, action, reward, next_state, done, truncated, env_props) -> float:
        ...
