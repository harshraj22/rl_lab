import numpy as np
import sys
import logging

sys.path.insert(0, '../')
from utils.utils import RunningMeanReinforce, softmax
from base.multi_arm_bandit_agent import MultiArmBanditAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReinforceAgent(MultiArmBanditAgent):

    def __init__(self, num_arms: int, baseline: bool = True, alpha: float = 0.3, beta: float = 0.3) -> None:
        """Initialize the ReinforceAgent. Each arm has a preference score, the
        agent selects an arm by sampling from the probability distribution defined
        by the softmax of the preferences. The baseline is used to estimate how
        good reward did the selected arm bring compared to the average reward that
        the agent has been recieved so far.

        Parameters
        ----------
        num_arms : int
            Number of arms in the bandit.
        baseline : bool, optional
            If the agent should use a baseline, by default True
        alpha : float, optional
            The rate at which the running average reward is to be updated by the
            agent, by default 0.3
            Read more in banditsComparision.pdf in 'lab2/ques/'
        beta: float, optional
            The rate at which the preference of the selected arm is to be updated
        """
        super(ReinforceAgent, self).__init__()
        self.num_arms = num_arms
        self.baseline = baseline
        self.running_means = [RunningMeanReinforce(baseline=self.baseline, beta=self.beta) for _ in range(self.num_arms)]
        self._baseline_rewards_mean = 0.0
        self.alpha = alpha
        self.beta = beta

    def reset(self) -> None:
        self.running_means = [RunningMeanReinforce(baseline=self.baseline, beta=self.beta) for _ in range(self.num_arms)]
        self._baseline_rewards_mean = 0.0

    @property
    def average_reward(self):
        return self._baseline_rewards_mean

    def update_mean(self, arm_index: int, reward: int) -> None:
        # update the running average reward, used for the baseline
        self._baseline_rewards_mean = (1 - self.alpha) * self._baseline_rewards_mean + self.alpha * reward
        # update the underlying preference of the selected arm
        self.running_means[arm_index].update_preference(reward, self.average_reward)

    def forward(self, state: int) -> int:
        """Choose the arm whose probability is maximum. The probability is calculated
        by taking softmax of the preferences of all arms."""
        pi = softmax([mean.preference for mean in self.running_means])
        # logger.info(f'pi: {pi}')
        return np.random.choice(self.num_arms, 1, p=pi)[0]

    def __call__(self, state: int) -> int:
        action = self.forward(state)
        return action

    def __str__(self):
        return f'ReinforceAgent(arms={self.num_arms}, baseline={self.baseline})'

    def __repr__(self) -> str:
        return self.__str__()
