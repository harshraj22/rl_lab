import numpy as np
import sys
import logging
from math import inf

sys.path.insert(0, '../')
from utils.utils import RunningMeanUCB
from base.multi_arm_bandit_agent import MultiArmBanditAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class UCBAgent(MultiArmBanditAgent):

    def __init__(self, num_arms: int) -> None:
        """UCB agent. The agent selects an action greedily with arm probabilities
        weighted by their empirical mean and the bonus term. the bonus term takes
        into account the current timestep and the number of times the current 
        arm has been played. The agent explores till each arm hasn't got enough
        number of chances and then exploits.

        Parameters
        ----------
        num_arms : int
            The number of arms in the Multi Arm Bandit environment.
        """
        super(UCBAgent, self).__init__()
        self.num_arms = num_arms
        self.running_means = [RunningMeanUCB() for _ in range(num_arms)]

    def update_mean(self, arm_index: int, reward: int) -> None:
        self.running_means[arm_index].update_mean(reward)

    def forward(self, state: int) -> int:
        """Select an action using the UCB over the estimated means and the bonus
        term."""
        return np.argmax([mean.priority for mean in self.running_means])

    def __call__(self, state: int) -> int:
        action = self.forward(state)
        for mean in self.running_means:
            mean.tick()
        return action

    def __str__(self) -> str:
        return f'UCBAgent(arms={self.num_arms})'

    def __repr__(self) -> str:
        return self.__str__()