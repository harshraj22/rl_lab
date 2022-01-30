import numpy as np
import sys
import logging
from math import inf

sys.path.insert(0, '../')
from utils.utils import RunningMeanThompson
from base.multi_arm_bandit_agent import MultiArmBanditAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ThompsonSamplingAgent(MultiArmBanditAgent):

    def __init__(self, num_arms: int) -> None:
        """Thompson Sampling Agent. The agent selects an action using a random
        draw from a Beta distribution. The distribution is updated each time the
        arm is selected. The initial exploration comes from the initialization
        of the Beta distribution with alpha = 1 and beta = 1. As the time progresses
        the agent explores and adjusts alpha and beta accordingly.

        Parameters
        ----------
        num_arms : int
            The number of arms in the Multi Arm Bandit environment.
        """
        super(ThompsonSamplingAgent, self).__init__()
        self.num_arms = num_arms
        self.running_means = [RunningMeanThompson() for _ in range(num_arms)]

    def update_mean(self, arm_index: int, reward: int) -> None:
        self.running_means[arm_index].update_mean(reward)

    def forward(self, state: int) -> int:
        """Select an action using a random draw from a Beta distribution."""
        return np.argmax([mean.priority for mean in self.running_means])

    def __call__(self, state: int) -> int:
        action = self.forward(state)
        return action
    
    def __str__(self) -> str:
        return f'ThompsonSamplingAgent(arms={self.num_arms})'
    
    def __repr__(self) -> str:
        return self.__str__()