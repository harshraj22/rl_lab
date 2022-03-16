from abc import ABC, abstractmethod
from typing import Union, List


class BaseAgent(ABC):
    """Abstract class for agents"""

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
    def learn(self, reward: Union[int, List[int]]) -> None:
        # ToDo: update the type of reward. It should also have info about states
        """Update the agent's knowledge.

        Parameters
        ----------
        reward : Union[int, List[int]]
            The reward received from the environment. In case of online learning
            agents, a reward is a single value. In case of offline learning agents,
            a reward is a list of rewards over the whole trajectory.
        """
        pass