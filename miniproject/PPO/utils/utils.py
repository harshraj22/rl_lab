import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
import gym


def calculate_advantage(rewards: npt.ArrayLike, values: npt.ArrayLike, dones: npt.ArrayLike, gae_gamma=0.99, gae_lambda=0.95) -> npt.ArrayLike:
    advantage = np.zeros_like(rewards)

    # rewards.shape: (batch_size)
    for t, _ in enumerate(rewards):
        discount, a_t = 1, 0
        # calculate advantage at timestep t
        for k in range(t, len(rewards)-1):
            a_t += discount * (rewards[k] + gae_gamma * values[k+1] * (1-int(dones[k])) - values[k])
            discount *= gae_gamma * gae_lambda

        advantage[t] = a_t
    
    return advantage


class TaxiEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def step(self, action):
        obs, reward, done, info = self.env.step(action.item())
        return [obs], reward, done, info
    
    def reset(self):
        return [self.env.reset()]