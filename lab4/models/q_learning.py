from typing import DefaultDict, List, Tuple, Union
import numpy as np

import sys
sys.path.append('..')

from base.baseagent import BaseAgent
from utils.utils import Sample


class QLearning(BaseAgent):
    """Agent that uses Q-Learning. (TD-0)
    
    Properties:
    ----------------
    Off Policy: The policy is not improved using samples generated from the current
        policy. More precisely, it tries to approximate the optimal policy using
        samples generated from the current policy.
    Online: Agent updates the policy after every step (taking an action and
        recieving a reward).
    TD-0: The Error is calculated considering only reward from the next timestamp.
        delta = R + gamma * Q(s', a') - Q(s, a)
    
    https://leimao.github.io/images/blog/2019-03-14-RL-On-Policy-VS-Off-Policy/q-learning.png
    """

    def __init__(self, num_states: int, num_actions: int, eps: float = 0.2,
        decay_factor: float = 0.99, lr: float = 0.2, gamma: float = 0.9) -> None:
        """
        Parameters:
        ----------
        num_states : int
            Number of states in the environment.
        num_actions : int
            Number of actions in the environment.
        eps : float
            Probability of selecting a random action.
        decay_factor : float
            The rate at which epsilon decays.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        """
        super(QLearning, self).__init__()

        self.num_states = num_states
        self.num_actions = num_actions

        self.eps = eps
        self.decay_factor = decay_factor
        self.lr = lr
        self.gamma = gamma

        # initialize the policy
        self.policy = np.random.randint(0, self.num_actions, size=self.num_states)

        # initialize the Q-values
        self.Q = np.random.randn(self.num_states, self.num_actions)

        self.mode = 'online'

    def forward(self, state: int) -> int:
        """Select an action."""
        if np.random.rand() < self.eps:
            return np.random.randint(0, self.num_actions)
        return self.policy[state]

    def learn(self) -> None:
        """Update the policy"""
        self.eps = np.clip(self.eps * self.decay_factor, 0.01, 1.0)

    def step(self, sample: Sample) -> None:
        """Update the agent's knowledge and policy correspondingly"""
        state, action, reward, next_state = sample
        self.Q[state][action] += self.lr * (reward + self.gamma * self.Q[next_state].max() - self.Q[state][action])

        # update the policy
        self.policy[state] = np.argmax(self.Q[state])