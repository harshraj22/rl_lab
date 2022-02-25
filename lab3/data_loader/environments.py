import gym
from gym.spaces import Discrete
from typing import Tuple
import sys


sys.path.insert(0, '../')
from base.iteration_env import IterationEnv


class GridWorldEnvironment(IterationEnv):
    """A generic class for GridWorld environment.
    __________________
    |0  |0  |0  |1    |
    |0  |0  |0  |-100 |
    |0  |XXX|0  |0    |
    |0  |0  |0  |0    |
    -------------------

    There are 15 possible states. Each state has a reward associated with it.
    There is a transition probability associated with each action. If the agent
    selects an action 'a', there is 80% chance of the agent moving to the next
    state corresponding to action 'a', and 10% each chance of the agent moving to
    the next state corresponding to action that is orthogonal to action 'a'.
    For example, if the agent decides to move North, there is a 80% chance of
    moving to the state in North direction, and a 10% chance of moving to the
    state in East direction and 10% chance of moving to the state in West direction.
    """

    def __init__(self) -> None:
        super(GridWorldEnvironment, self).__init__()
        self.action_space = Discrete(4)
        self.observation_space = Discrete(15)

    
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