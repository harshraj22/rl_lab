from abc import ABC, abstractmethod
from typing import Any
import gym


class IterationAgent(ABC):
    """Abstract class for an iteration agent."""

    @abstractmethod
    def learn(self, env: gym.Env) -> Any:
        """Learn from the environment."""
        pass

    @abstractmethod
    def action(self, state: int) -> int:
        """Select an action."""
        pass