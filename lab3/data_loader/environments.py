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

    There are 15 possible states: [0, 14]. Each state has a reward associated with it.
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
        self._state = 0

    def reset(self) -> None:
        self._state = 0

    def render(self) -> None:
        pass

    def integer_state_to_coordinates(self, cur_state: int) -> Tuple[int, int]:
        """Convert the integer state to coordinates."""
        if cur_state >= 9:
            cur_state += 1
        x, y = cur_state // 4, cur_state % 4
        return x, y

    def coordinates_to_integer_state(self, x: int, y: int) -> int:
        """Convert the coordinates to integer state."""
        cur_state = x * 4 + y
        if cur_state >= 9:
            cur_state -= 1
        return cur_state

    def is_valid_coordinate(self, x: int, y: int) -> bool:
        """Check if the coordinates are valid."""
        return 0 <= x < 4 and 0 <= y < 4 and (x, y) != (2, 1)

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Return observation, reward, done, and info."""
        pass

    @property
    def state(self) -> int:
        """Return the current state of the environment."""
        return self._state

    @state.setter
    def state(self, new_state: int) -> None:
        """Set the current state of the environment."""
        self._state = new_state
