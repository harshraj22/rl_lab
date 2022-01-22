from abc import ABC, abstractmethod


class MultiArmBanditAgent:

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
        """
        pass

    @abstractmethod
    def update_mean(self, action: int, reward: int) -> None:
        """Update the running mean of the selected action."""
        pass

    @abstractmethod
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
        pass