import numpy as np
import sys
from typing import List
import logging
from math import inf

sys.path.insert(0, '../')
from utils.utils import RunningMean

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def softmax(x: List[float]) -> List[float]:
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class SoftmaxAgent:

    def __init__(self, num_arms: int, initial_temp: int = 1000, decay_factor: float=0.9) -> None:
        """Softmax agent. The agent selects an action greedily with arm probabilities
        equal to the softmax applied to corresponding estimated means divided by the
        tempreature. The exploration part comes with high tempreature, where the 
        actions are equiprobable. The tempreature is gradually annealed, reducing 
        by the decay factor, and allowing the agent to exploit more using the new
        estimated means.

        Parameters
        ----------
        num_arms : int
            The number of arms present in the Multi Arm Bandit environment.
        initial_temp : int, optional
            Initial tempreature, by default 1000
        decay_factor : float, optional
            The multiplicant to be multiplied with the temprature each time the
            agent selects an action, by default 0.9
            T_new = T_old * decay_factor
        """
        self.num_arms = num_arms
        self.current_temp = initial_temp
        self.decay_factor = decay_factor
        self.running_means = [RunningMean() for _ in range(num_arms)]

    def update_mean(self, action: int, reward: int) -> None:
        """Update the running mean of the selected action."""
        self.running_means[action].update_mean(reward)

    def forward(self, state: int) -> int:
        """Select an action using the softmax over the estimated means and the
        current tempreature.

        Parameters
        ----------
        state : int
            The current state of the environment.

        Returns
        -------
        int
            The index of the arm selected.
        """
        arm_selection_probs = softmax([mean.mean / self.current_temp for mean in self.running_means])
        action = np.random.choice(self.num_arms, p=arm_selection_probs)
        return action

    def __call__(self, state: int) -> int:
        self.current_temp = np.clip(self.current_temp * self.decay_factor, 0.001, inf)
        action = self.forward(state)
        return action