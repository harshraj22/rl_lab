import gym
from gym.spaces import Discrete

import numpy as np


class LinearEnv(gym.Env):
    """A generic class for a very simple Linear environment.
    
    [0 -> 1 -> 2 -> 3 -> 4 -> 5]
    Agent starts at state 0, and can go either forward or backward.
    State 5 is the terminal state and agent recieves a reward of 1 on reaching
    that state. An episode can be played for a maximum of _max_time steps.
    """
    def __init__(self, max_time: int = 10):
        super(LinearEnv, self).__init__()
        self.action_space = Discrete(2)
        self.observation_space = Discrete(6)
        self._state = 0
        self._time = 0
        self._max_time = max_time

    def reset(self):
        self._time = 1
        self._state = 0
        return self._state

    def step(self, action):
        self._time += 1
        action = -1 if not action else action
        new_state = np.clip(self._state + action, 0, 5)
        reward = 1 if new_state == 5 else 0
        done = True if self._time >= self._max_time or new_state == 5 else False
        self._state = new_state
        return new_state, reward, done, {}

    def render(self):
        state = ['_' for _ in range(self.observation_space.n)]
        state[self._state] = 'X'
        print(' '.join(state))