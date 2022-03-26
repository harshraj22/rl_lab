import gym
from gym.spaces import Discrete
import numpy as np
from typing import Tuple


class LinearEnvWrapper(gym.Wrapper):
    """.
    A wrapper for the LinearEnv environment that adds the following:
    - Boost up the reward recieved on reaching the terminal state.
    - Remove the sparsity of rewards, by adding the eucledian distance between
        the current and goal state as intermediate rewards.
    """
    def __init__(self, env) -> None:
        super().__init__(env)
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._time = 1

    def step(self, action: int):
        cur_state = self.env.state
        state, reward, done, info = self.env.step(action)
        self._time += 1
        reward = reward * 100 + (state - cur_state)
        return state, reward, done, info


class MountainCarEnvWrapper(gym.Wrapper):
    """A wrapper for the MountainCar-v0 environment that adds the following:
    Convert the continious state, actions into discrete bins.

    Divides action space and observation space into n_bins bins each. each
    (action, observation) tuple is mapped to a unique integer representing the
    new state.
    """
    def __init__(self, env, n_bins: int = 10) -> None:
        super().__init__(env)

        self.env = env
        self.n_bins = n_bins
        self.position_bins = np.linspace(-1.2, 0.07, self.n_bins)
        self.velocity_bins = np.linspace(0.6, 0.07, self.n_bins)
        self.observation_space = Discrete(self.n_bins * self.n_bins)

        # state = position_bin * n_bins + velocity_bin
    
    def get_digitized_state(self, state: Tuple[float, float]) -> int:
        """Returns the digitized state."""
        position, velocity = state
        position_bin = np.digitize(position, self.position_bins)
        velocity_bin = np.digitize(velocity, self.velocity_bins)

        new_state = position_bin * self.n_bins + velocity_bin
        return int(new_state)

    def step(self, action: int):
        state, reward, done, info = self.env.step(action)
        return self.get_digitized_state(state), reward, done, info

    def reset(self) -> int:
        return self.get_digitized_state(self.env.reset())