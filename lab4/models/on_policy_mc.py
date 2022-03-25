from collections import defaultdict
from typing import DefaultDict, List, Tuple, Deque
import numpy as np
from collections import deque

import sys
sys.path.append('..')

from base.baseagent import BaseAgent
from utils.utils import Sample


class FirstVisitMonteCarlo(BaseAgent):
    """Agent that uses first-visit Monte Carlo (on-policy MC).

    Properties:
    ----------------
    First Visit: Use rewards from the first time the state-action pair is visited in
        a trajectory.
    On Policy: The policy is improved using samples generated from the current
        policy.
    Offline: The agent has to wait till the end of the trajectory to update the
        policy.
    https://i.stack.imgur.com/033M8.png
    """
    def __init__(self, num_states: int, num_actions: int, eps: float = 0.2, decay_factor: float = 0.9) -> None:
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
        """
        super(FirstVisitMonteCarlo, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.eps = eps
        self.decay_factor = decay_factor

        # initialize the policy
        self.policy = np.random.randint(0, self.num_actions, size=self.num_states)

        # initialize the Q-values
        self.Q = np.random.randn(self.num_states, self.num_actions)

        # Returns
        self.returns: DefaultDict[Tuple[int, int], Deque[int]] = defaultdict(lambda: deque())

        self.mode = 'offline'

    def forward(self, state: int) -> int:
        """Select an action."""
        if np.random.rand() < self.eps:
            return np.random.randint(0, self.num_actions)
        return self.policy[state]

    def learn(self) -> None:
        """Update the policy"""
        for state in range(self.num_states):
            for action in range(self.num_actions):
                self.Q[state][action] = np.mean(self.returns[(state, action)]) if self.returns[(state, action)] else 0

        # update the policy
        for state in range(self.num_states):
            self.policy[state] = np.argmax(self.Q[state])

        self.eps = np.clip(self.eps * self.decay_factor, 0.01, 1)

    def step(self, sample: Sample) -> None:
        """Update the agent's knowledge.

        Parameters
        ----------
        sample: info about a timestep of the trajectory
        """
        state, action, return_, next_state = sample
        # update the returns
        # if (state, action) in self.returns.keys():
        #     pass # skip if the state-action pair has been visited before
        self.returns[(state, action)].append(return_)
