import numpy as np
import sys

sys.path.insert(0, '../')
from utils.utils import RunningMean


class EpsilonGreedyAgent:

    def __init__(self, eps: float, num_arms: int, initial_temp: int = None, decay_factor: float=1.0) -> None:
        """Epsilon greedy agent. The agent selects an action randomly with probability eps,
        and greedily selects the action with the highest expected reward with probability 1-eps.
        The agent also has a temprature parameter that controls the exploration vs exploitation
        tradeoff. The temperature decays with time, reducing the espilon gradually.

        Parameters
        ----------
        eps : float
            The probability of selecting a random action.
        num_arms : int
            The number of arms present in the Multi Arm Bandit environment.
        initial_temp : int, optional
            Initial tempreature, by default None
        decay_factor : float, optional
            The multiplicant to be multiplied with the temprature each time the
            agent selects an action, by default 1.0
        """
        self.eps = eps
        self.num_arms = num_arms
        self.current_temp = initial_temp
        self.decay_factor = decay_factor
        self.running_means = [RunningMean() for _ in range(num_arms)]

    def update_mean(self, action: int, reward: int) -> None:
        """Update the running mean of the selected action."""
        self.running_means[action].update_mean(reward)

    def forward(self, state: int, eps: float) -> int:
        """Select an action.

        Parameters
        ----------
        state : int
            The current state of the environment.
        eps : float
            The probability of selecting a random action.

        Returns
        -------
        int
            The index of the arm selected.
        """
        if np.random.random() < eps:
            return np.random.choice(self.num_arms)
        else:
            return np.argmax([mean.mean for mean in self.running_means])

    def __call__(self, state: int) -> int:

        # if it is a variable epsilon agent, update the temperature
        if self.current_temp is not None:
            self.current_temp *= self.decay_factor
            eps = self.eps / self.current_temp
        else:
            eps = self.eps

        action = self.forward(state, eps)

        return action
