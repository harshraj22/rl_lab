
import numpy as np
from math import inf
from typing import List


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
        else:
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


class RunningMeanThompson(RunningMean):
    """Class to store and update the alpha and beta values for the Thompson
    Sampling using Beta distribution. The reward distribution is assumed to be
    binomial."""
    
    def __init__(self):
        super(RunningMeanThompson, self).__init__()
        """ Set the initial alpha and beta values to 1. These correspond to the
        arguments of the beta distribution. Initial value of 1 is chosen to
        set the initial distribution as uniform, ie. no prior knowledge about
        the underlying reward distribution.  """
        self.alpha = 1
        self.beta = 1

    def update_mean(self, reward: int) -> None:
        self.alpha += int(reward == 1)
        self.beta += int(reward == 0)

        super(RunningMeanThompson, self).update_mean(reward)

    @property
    def priority(self) -> float:
        return np.random.beta(self.alpha, self.beta)


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
        return self._preference

    def update_mean(self, reward: int) -> None:
        # self.running_mean_reward = (1-self.alpha) * self.running_mean_reward + self.alpha * reward
        self.running_mean_reward += reward
        self.times_selected += 1

    def update_preference(self, reward: int) -> None:
        self._preference = self._preference + self.beta * (reward - (self.mean if self.baseline else 0))
        self._preference = np.clip(self._preference, 0, 200)

    def __str__(self) -> str:
        return f'Mean: {self.mean:.3f}, Preference: {self.preference:.3f}'

    def __repr__(self) -> str:
        return self.__str__()
    