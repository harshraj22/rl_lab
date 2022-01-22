
import numpy as np
from math import inf


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

