

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
        return f'Mean: {self.mean}'
        