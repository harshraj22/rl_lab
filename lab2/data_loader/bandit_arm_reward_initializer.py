import imp
import sys
from typing import Tuple, List


sys.path.insert(0, '../')
from data_loader.probablistic_reward_distributions import BinomialRewardDistribution
from base.reward_distribution import RewardDistribution
from base.arm_reward_initilizer import ArmRewardInitilizer



class BanditArmRewardInitializer(ArmRewardInitilizer):

    def __init__(self, initializer_type: str) -> None:
        super(ArmRewardInitilizer, self).__init__()
        self.initializer_type = initializer_type

    
    def __call__(self, num_arms: int=3) -> Tuple[List[RewardDistribution], int]:
        """Initialize the reward distributions for each arm.
        
        Parameters
        ----------
        num_arms: int
            The number of arms in our multi-armed bandit.

        Returns
        -------
        Tuple[List[RewardDistribution], int]: A tuple of reward distributions and the
            index of the optimal arm.
        """
        if self.initializer_type == "binomial":
            return [BinomialRewardDistribution(1.0/num_arms) for _ in range(num_arms)], 0
        else:
            raise ValueError(f"Initializer not found.") 


