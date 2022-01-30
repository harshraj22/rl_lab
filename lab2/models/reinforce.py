import numpy as np
import sys
import logging
from math import inf

sys.path.insert(0, '../')
from utils.utils import RunningMeanReinforce, softmax
from base.multi_arm_bandit_agent import MultiArmBanditAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReinforceAgent(MultiArmBanditAgent):

    def __init__(self, num_arms: int) -> None:
        super(ReinforceAgent, self).__init__()
        self.num_arms = num_arms
        self.running_means = [RunningMeanReinforce() for _ in range(num_arms)]

    def update_mean(self, arm_index: int, reward: int) -> None:
        self.running_means[arm_index].update_preference(reward)
        self.running_means[arm_index].update_mean(reward)

    def forward(self, state: int) -> int:
        """Choose the arm whose probability is maximum. The probability is calculated
        by taking softmax of the preferences of all arms."""
        pi = softmax([mean.preference for mean in self.running_means])
        return np.argmax(pi)

    def __call__(self, state: int) -> int:
        action = self.forward(state)
        return action
    