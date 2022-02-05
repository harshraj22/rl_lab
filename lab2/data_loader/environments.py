import gym
from gym.spaces import Discrete
import sys
from typing import Tuple, Callable


class MultiArmBanditEnvironment(gym.Env):
    """A generic class for MultiArmBandit environments. The underlying reward
    for each arm is initialized by a function. The simulation can be run for a
    fixed number of timesteps."""

    def __init__(self, arm_initializer: Callable, num_arms: int=3, total_timesteps: int=1000) -> None:
        super(MultiArmBanditEnvironment, self).__init__()

        self.num_arms = num_arms
        self.reward_distributions, self.optimal_arm_index, self._optimal_mean = arm_initializer(num_arms)
        self.total_timesteps = total_timesteps
        self.current_timestep = 0
        self.total_optimal_arms_hits = 0

        self.action_space = Discrete(num_arms)
        self.observation_space = Discrete(total_timesteps)

    def __str__(self) -> str:
        return f'Env: {self.reward_distributions}'

    @property
    def optimal_mean(self) -> float:
        return self._optimal_mean

    def reset(self) -> int:
        self.current_timestep = 0
        self.total_optimal_arms_hits = 0
        return self.current_timestep

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Return observation, reward, done, and info."""
        reward = self.reward_distributions[action].sample()
        self.current_timestep += 1
        done = self.current_timestep >= self.total_timesteps
        self.total_optimal_arms_hits += 1 if action == self.optimal_arm_index else 0
        return self.current_timestep, reward, done, {'optimal_arm_hits': self.total_optimal_arms_hits}

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass