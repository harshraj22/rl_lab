"""Module for storing the gym environments for simulation"""

import gym
import numpy as np
from gym import spaces


class SnakeAndLadderEnv(gym.Env):
    """Gym Environment to simulate the snake and ladder game"""
    def __init__(self, current_state=0, num_states=9, dead_states=(2, 4), view=None) -> None:
        """Initialize the environment. The player starts at position 'current_state',
        and takes actions in range(1, 3). The game ends when the player reaches 
        either the 'dead_states' (in which case he/she looses and gets a reward
        of 0) or when he/she reaches the 'num_states-1' postion (in which case
        he/she wins and gets a reward of 1).

        Args:
            current_state (int, optional): the starting state of the player. Defaults to 0.
            num_states (int, optional): the total number of states. Defaults to 9.
            dead_states (tuple, optional): the dead states. Defaults to (2, 4).
            view (tuple, optional): In order to render the environment, the shape
              in which the environment should be printed. Note that the 'num_states'
              should be equal to the product of elements of 'view'. If not provided,
              it is assumed to be of shape (num_states,).
        """
        super().__init__()
        self.current_state = current_state
        self.num_states = num_states
        self.dead_states = dead_states
        self.view = view if view else (self.num_states, )

    def reset(self):
        self.current_state = 0
        return self.current_state

    def step(self, action):
        # you can not step out of the board by taking an action. Such actions would be ignored.
        if self.current_state + action < self.num_states:
            self.current_state += action
        return (
            self.current_state,
            int(self.current_state == self.num_states - 1),
            bool(self.current_state in [self.num_states-1, *self.dead_states]),
            {}
        )
   
    def render(self):
        states = np.chararray(shape=(self.num_states,), unicode=True)
        for i in range(self.num_states):
            states[i] = str(i)
        states[self.current_state] = '#'
        states[list(self.dead_states)] = '_'
        print(states.reshape(self.view))
        print('#: current state \n_: dead state\n')

    def close(self):
        pass
