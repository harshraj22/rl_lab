from abc import ABC, abstractmethod


class RewardDistribution(ABC):

    @abstractmethod
    def sample(self) -> float:
        """Sample the reward from the underlying distribution.

        Returns
        -------
        float: The sampled reward.
        """
        pass
