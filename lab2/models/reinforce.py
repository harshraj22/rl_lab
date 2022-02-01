import numpy as np
import sys
import logging

sys.path.insert(0, '../')
from utils.utils import RunningMeanReinforce, softmax
from base.multi_arm_bandit_agent import MultiArmBanditAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReinforceAgent(MultiArmBanditAgent):

    def __init__(self, num_arms: int, baseline: bool = True) -> None:
        super(ReinforceAgent, self).__init__()
        self.num_arms = num_arms
        self.baseline = baseline
        self.running_means = [RunningMeanReinforce(baseline=baseline) for _ in range(num_arms)]
        self._baseline_rewards = 0.0
        self._timestep = 0

    @property
    def average_reward(self):
        if self._timestep == 0:
            return 0.0
        return self._baseline_rewards / self._timestep

    def update_mean(self, arm_index: int, reward: int) -> None:
        self.running_means[arm_index].update_preference(reward, self.average_reward)

    def forward(self, state: int) -> int:
        """Choose the arm whose probability is maximum. The probability is calculated
        by taking softmax of the preferences of all arms."""
        pi = softmax([mean.preference for mean in self.running_means])
        # logger.info(f'pi: {pi}, sum: {np.sum(pi)}')
        self._timestep += 1
        return np.random.choice(self.num_arms, 1, p=pi)[0]

    def __call__(self, state: int) -> int:
        action = self.forward(state)
        return action

    def __str__(self):
        return f'ReinforceAgent(arms={self.num_arms}, baseline={self.baseline})'

    def __repr__(self) -> str:
        return self.__str__()
