import sys
import numpy as np

sys.path.insert(0, '..')
from base.reward_distribution import RewardDistribution



class BinomialRewardDistribution(RewardDistribution):
    """A binomial distribution with probability parameter p."""

    def __init__(self, p: float) -> None:
        super(BinomialRewardDistribution, self).__init__()
        self.p = p

    def sample(self) -> float:
        """Returns reward of 1 with probability p and reward of 0 with probability 1-p."""
        return np.random.binomial(1, self.p)
