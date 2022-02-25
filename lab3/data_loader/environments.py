import gym
from gym.spaces import Discrete
from typing import Tuple
import sys


sys.path.insert(0, '../')
from base.iteration_env import IterationEnv


class GridWorldEnvironment(IterationEnv):
    """A generic class for GridWorld environment."""

    def __init__(self) -> None:
        super(GridWorldEnvironment, self).__init__()

    
    def reset(self) -> None:
        pass

    def render(self) -> None:
        pass

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Return observation, reward, done, and info."""
        pass

    @property
    def state(self) -> int:
        """Return the current state of the environment."""
        pass

    @state.setter
    def state(self, state: int) -> None:
        """Set the current state of the environment."""
        pass