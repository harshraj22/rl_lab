from abc import ABC, abstractmethod
from typing import Union, List

import sys
sys.path.append('..')
from utils.utils import Sample


class BaseAgent(ABC):
    """Abstract class for agents"""
    mode: str = 'offline'

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
            Action to select.
        """
        pass

    def __call__(self, state: int) -> int:
        return self.forward(state)

    @abstractmethod
    def step(self, sample: Sample) -> None:
        """Update the agent's knowledge.

        Parameters
        ----------
        sample: info about a timestep of the trajectory
        """
        pass

    @abstractmethod
    def learn(self) -> None:
        """Update the agent's policy."""
        pass