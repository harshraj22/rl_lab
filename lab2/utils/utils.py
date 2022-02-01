import numpy as np
from math import inf
from typing import List
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def softmax(x: List[float]) -> List[float]:
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class RunningMean:
    """Class to store and update the running mean."""
    def __init__(self):
        self.total_reward = 0.0
        self.count = 0

    @property
    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_reward / self.count

    def update_mean(self, reward: int) -> None:
        self.total_reward += reward
        self.count += 1

    def __str__(self) -> str:
        return f'Mean: {self.mean:.3f}'

    def __repr__(self) -> str:
        return self.__str__()


class RunningMeanUCB(RunningMean):
    """Class to store and update the running mean and bonus term for the UCB
    agent. As the bonus term needs information about the time passed, the agent
    need to tick() every time it takes an action, irrespective of whether the
    arm corresponding to this object is chosen or not."""
    def __init__(self):
        super(RunningMeanUCB, self).__init__()
        self.bonus = inf
        self.time = 0

    @property
    def priority(self) -> float:
        return self.mean + self.bonus

    def tick(self):
        self.time += 1
        if self.count > 0:
            self.bonus = np.sqrt(2 * np.log(self.time) / self.count)


class RunningMeanThompsonBeta(RunningMean):
    """Class to store and update the alpha and beta values for the Thompson
    Sampling using Beta distribution. The reward distribution is assumed to be
    binomial."""

    def __init__(self):
        super(RunningMeanThompsonBeta, self).__init__()
        """ Set the initial alpha and beta values to 1. These correspond to the
        arguments of the beta distribution. Initial value of 1 is chosen to
        set the initial distribution as uniform, ie. no prior knowledge about
        the underlying reward distribution.  """
        self.alpha = 1
        self.beta = 1

    def update_mean(self, reward: int) -> None:
        self.alpha += int(reward == 1)
        self.beta += int(reward == 0)

        super(RunningMeanThompsonBeta, self).update_mean(reward)

    @property
    def priority(self) -> float:
        return np.random.beta(self.alpha, self.beta)


class RunningMeanThompsonGaussian(RunningMean):
    def __init__(self):
        super(RunningMeanThompsonGaussian, self).__init__()
        self.mu = 0
        self.sigma = 1
        self.rewards = [0]

    @property
    def mean(self) -> float:
        return self.mu

    def update_mean(self, reward: int) -> None:
        # https://stackoverflow.com/a/50729600/10127204
        # N = self.count + 1
        # rho = 1.0 / N
        # d = reward - self.mu
        # self.mu += rho * d
        # self.sigma += rho * ((1-rho)* d**2 - self.sigma)
        # self.sigma = np.clip(self.sigma, 0.001, 10_000)
        self.rewards.append(reward)
        super(RunningMeanThompsonGaussian, self).update_mean(reward)

    @property
    def priority(self) -> float:
        # Assuming initial mean=0, sigma=1
        # Following point 2 from bayesNormal.pdf
        initial_mean, initial_std = 0, 1
        self.mu = np.mean(self.rewards)
        self.sigma = np.std(self.rewards)

        std_sq = 1.0 / np.clip((1.0 / initial_std**2 + self.count / self.sigma**2), 0.001, 10_000)
        new_mean = std_sq * (initial_mean / initial_std**2 + self.mean*self.count / self.sigma**2)
        # logger.info(f'Mean: {new_mean:.3f}, std: {np.sqrt(std_sq):.3f} | {self.mu:.3f}, {self.sigma:.3f}')
        return np.random.normal(new_mean, np.sqrt(std_sq), 1).item()
        return np.random.normal(self.mu, self.sigma)


class RunningMeanReinforce(RunningMean):
    """Class to store running mean and preference for the REINFORCE agent."""
    def __init__(self, alpha: float = 0.8, beta: float = 0.3, baseline: bool = True):
        """Initialize the class. The agent choses the arm with the highest preference.
        The preference is updated every time the agent selects the corresponding
        arm and recieves a reward. Running mean is maintained to implement the
        baseline.

        Parameters
        ----------
        alpha : float
            Rate at which the running mean of rewrads is updated.
        beta : float
            Rate at which the preference is updated. Low beta will be more robust
            to outliers, but will take more number of iterations to converge.
        baseline : bool
            Whether to use the baseline or not.
        """
        super(RunningMeanReinforce, self).__init__()

        self.running_mean_reward = 0
        self.alpha = alpha
        self.beta = beta
        self._preference = 0.0
        self.baseline = baseline
        self.times_selected = 1

    @property
    def mean(self) -> float:
        return self.running_mean_reward / self.times_selected

    @property
    def preference(self) -> float:
        """Denotes the preference that should be used for the current arm, while
        selecting the arm."""
        return self._preference

    def update_mean(self, reward: int, average_reward: float = 0.0) -> None:
        self.running_mean_reward = (1-self.alpha) * average_reward + self.alpha * reward
        # self.running_mean_reward += reward
        # self.times_selected += 1

    def update_preference(self, reward: int, average_reward: float = 0.0) -> None:
        self._preference = self._preference + self.beta * (reward - (average_reward if self.baseline else 0))
        self._preference = np.clip(self._preference, 0, 700)

    def __str__(self) -> str:
        return f'Mean: {self.mean:.3f}, Preference: {self.preference:.3f}'

    def __repr__(self) -> str:
        return self.__str__()
    