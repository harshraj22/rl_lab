from collections import defaultdict
from typing import DefaultDict, List, Tuple
import numpy as np

import sys
sys.path.append('..')

from base.baseagent import BaseAgent


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
    def __init__(self, num_states: int, num_actions: int, eps: float = 0.2) -> None:
        super(FirstVisitMonteCarlo, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.eps = eps

        # initialize the policy
        self.policy = np.random.randint(0, self.num_actions, size=self.num_states)

        # initialize the Q-values
        self.Q = np.random.randn(self.num_states, self.num_actions)

        # Returns
        self.returns: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)

    def forward(self, state: int) -> int:
        """Select an action."""
        return self.policy[state]

    def learn(self) -> None:
        """Update the policy"""
        for state in range(self.num_states):
            for action in range(self.num_actions):
                self.Q[state][action] = np.mean(self.returns[(state, action)])

        # update the policy
        for state in range(self.num_states):
            self.policy[state] = np.argmax(self.Q[state])

    def step(self, state: int, action: int, reward: int) -> None:
        """Update the agent's knowledge.

        Parameters
        ----------
        state : int
            The current state of the environment.
        action : int
            The action selected by the agent.
        reward : int
            The reward received from the environment.
        """
        # update the returns
        if (state, action) in self.returns.keys():
            pass # skip if the state-action pair has been visited before
        self.returns[(state, action)].append(reward)