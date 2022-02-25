import gym
from abc import ABC, abstractmethod


class IterationEnv(ABC, gym.Env):
    """Abstract class for an iteration environment."""

    @property
    @abstractmethod
    def state(self) -> int:
        """Return the current state of the environment."""
        pass

    @state.setter
    @abstractmethod
    def state(self, state: int) -> None:
        """Set the current state of the environment."""
        pass