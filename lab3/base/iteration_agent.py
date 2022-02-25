from abc import ABC, abstractmethod
import gym


class IterationAgent(ABC):
    """Abstract class for an iteration agent."""

    @abstractmethod
    def learn(self, env: gym.Env) -> None:
        """Learn from the environment."""
        pass

    @abstractmethod
    def action(self, state: int) -> int:
        """Select an action."""
        pass