import gym
from gym.spaces import Discrete
from typing import Tuple
import sys
import numpy as np


sys.path.insert(0, '../')
from base.iteration_env import IterationEnv

from enum import IntEnum


class Directions(IntEnum):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3

    TOTAL_DIRECTIONS = 4


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

    def __init__(self, start_state: int = 0, gamma: float = 0.9) -> None:
        super(GridWorldEnvironment, self).__init__(gamma)
        self.action_space = Discrete(4)
        self.observation_space = Discrete(15)
        self._state = start_state
        self._goal_state = 3
        self._terminal_state = 7
        self._goal_state_reward = 1
        self._terminal_state_reward = -100

    def reset(self, start_state: int = 0) -> None:
        self._state = start_state

    def render(self) -> None:
        pass

    def transition_probability(self, action: int, initial_state: int, final_state: int) -> float:
        """Return the transition probability of ending at final_state if taken
        action in initial_state."""
        if initial_state == self._terminal_state or initial_state == self._goal_state:
            return initial_state == final_state

        x_i, y_i = self.integer_state_to_coordinates(initial_state)
        x_f, y_f = self.integer_state_to_coordinates(final_state)

        required_probability = 0.0
        action_dict = {
            Directions.UP: (-1, 0),
            Directions.DOWN: (1, 0),
            Directions.RIGHT: (0, 1),
            Directions.LEFT: (0, -1)
        }
        # update (x_i, y_i) corresponding to the action with taking the boundary
        # of grid into account
        x_i_final, y_i_final = x_i, y_i
        if self.is_valid_coordinate(x_i + action_dict[action][0], y_i + action_dict[action][1]):
            x_i_final, y_i_final = x_i + action_dict[action][0], y_i + action_dict[action][1]

        # if (x_i_final, y_i_final) == (x_f, y_f): add 0.8 to the probability
        if (x_i_final, y_i_final) == (x_f, y_f):
            required_probability = 0.8

        # do the same with perpendicular action, adding 0.1 to the probability
        # on success
        perpendicular_action = (action + 1) % Directions.TOTAL_DIRECTIONS
        x_i_final, y_i_final = x_i, y_i
        if self.is_valid_coordinate(x_i + action_dict[perpendicular_action][0], y_i + action_dict[perpendicular_action][1]):
            x_i_final, y_i_final = x_i + action_dict[perpendicular_action][0], y_i + action_dict[perpendicular_action][1]

        # if (x_i_final, y_i_final) == (x_f, y_f): add 0.8 to the probability
        if (x_i_final, y_i_final) == (x_f, y_f):
            required_probability += 0.1

        perpendicular_action = (action - 1 + Directions.TOTAL_DIRECTIONS) % Directions.TOTAL_DIRECTIONS
        x_i_final, y_i_final = x_i, y_i
        if self.is_valid_coordinate(x_i + action_dict[perpendicular_action][0], y_i + action_dict[perpendicular_action][1]):
            x_i_final, y_i_final = x_i + action_dict[perpendicular_action][0], y_i + action_dict[perpendicular_action][1]

        # if (x_i_final, y_i_final) == (x_f, y_f): add 0.8 to the probability
        if (x_i_final, y_i_final) == (x_f, y_f):
            required_probability += 0.1


        # return the probability
        return required_probability


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
        if self._state == self._terminal_state or self._state == self._goal_state:
            return self._state, 0, True, {}

        x, y = self.integer_state_to_coordinates(self._state)

        transition_probability = np.random.uniform(0, 1)
        if 0.8 <= transition_probability < 0.9:
            action = (action - 1 + Directions.TOTAL_DIRECTIONS) % Directions.TOTAL_DIRECTIONS
        elif transition_probability >= 0.9:
            action = (action + 1 + Directions.TOTAL_DIRECTIONS) % Directions.TOTAL_DIRECTIONS

        if action == Directions.UP:
            y = y - 1
        elif action == Directions.DOWN:
            y = y + 1
        elif action == Directions.RIGHT:
            x = x + 1
        elif action == Directions.LEFT:
            x = x - 1
        else:
            raise ValueError("Illegal action")

        if self.is_valid_coordinate(x, y):
            self._state = self.coordinates_to_integer_state(x, y)

        reward, done = 0, False
        if self._state == self._goal_state:
            reward, done = self._goal_state_reward, True
        elif self._state == self._terminal_state:
            reward, done = self._terminal_state_reward, True
        
        return self._state, reward, done, {}        

    @property
    def state(self) -> int:
        """Return the current state of the environment."""
        return self._state

    @state.setter
    def state(self, new_state: int) -> None:
        """Set the current state of the environment."""
        self._state = new_state


if __name__ == '__main__':
    pass
    # env = GridWorldEnvironment()
    # done = False
    # while not done:
    #     action = env.action_space.sample()
    #     print(action)
    #     obs, reward, done, _ = env.step(action)
    #     print(obs, reward, done)