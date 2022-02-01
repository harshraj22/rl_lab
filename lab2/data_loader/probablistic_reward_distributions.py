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


class GaussianRewardDistribution(RewardDistribution):
    """A Gaussian distribution with mean mu and standard deviation sigma."""

    def __init__(self, mu: float, sigma: float) -> None:
        super(GaussianRewardDistribution, self).__init__()
        self.mu = mu
        self.sigma = sigma

    def sample(self) -> float:
        """Returns a sample from the Gaussian distribution."""
        return np.random.normal(self.mu, self.sigma)
    
    def __str__(self) -> str:
        return f'Gaussian(mu={self.mu}, sigma={self.sigma})'

    def __repr__(self) -> str:
        return self.__str__()
