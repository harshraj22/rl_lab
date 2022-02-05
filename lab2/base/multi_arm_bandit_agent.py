from abc import ABC, abstractmethod


class MultiArmBanditAgent(ABC):

    @abstractmethod
    def update_mean(self, action: int, reward: int) -> None:
        """Update the running mean of the selected action."""
        pass

    def reset(self):
        """Reset the agent initialization"""
        pass

    @abstractmethod
    def forward(self, state: int) -> int:
        """Select an action.

        Parameters
        ----------
        state : int
            The current state of the environment.

        Returns
        -------
        int
            The index of the arm selected.
        """
        pass

    def __call__(self, state: int) -> int:
        """Select an action.

        Parameters
        ----------
        state : int
            The current state of the environment.

        Returns
        -------
        int
            The index of the arm selected.
        """
        return self.forward(state)
