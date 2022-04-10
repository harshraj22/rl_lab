from typing import DefaultDict, List, Tuple, Union
import numpy as np

import sys
sys.path.append('..')

from base.baseagent import BaseAgent
from utils.utils import Sample
from utils.tile import TiledQTable
from utils.features import Phi


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
        delta = R + gamma * max_over_a'{ Q(s', a') } - Q(s, a)

    https://leimao.github.io/images/blog/2019-03-14-RL-On-Policy-VS-Off-Policy/q-learning.png
    """

    def __init__(self, states_range: List[Tuple[float, float]], num_actions: int, tiling_specs, feat_dim: int = 5, eps: float = 0.2,
        decay_factor: float = 0.99, lr: float = 0.2, gamma: float = 0.9) -> None:
        """
        Parameters:
        ----------
        states_range : array like
            Lower bounds for each dimension of state space.
        num_actions : int
            Number of actions in the environment.
        feat_dim: int
            Number of dimentions in features for each state.
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

        self.states_range = states_range
        self.num_actions = num_actions
        self.tiling_specs = tiling_specs
        self.feat_dim = feat_dim
        self.phi = Phi(self.feat_dim)

        self.eps = eps
        self.decay_factor = decay_factor
        self.lr = lr
        self.gamma = gamma

        # initialize the policy
        # self.policy = np.random.randint(0, self.num_actions, size=self.num_states)

        self.W = np.random.randn(self.feat_dim)

        # initialize the Q-values
        self.Q = TiledQTable(self.states_range[0], self.states_range[1], self.tiling_specs, self.num_actions)

        self.mode = 'online'

    def forward(self, state) -> int:
        """Select an action."""
        if np.random.rand() < self.eps:
            return np.random.randint(0, self.num_actions)
        # return self.policy[state]
        return int(np.argmax([self.Q.get(state, action) for action in range(self.num_actions)]))

    def learn(self) -> None:
        """Update the policy"""
        self.eps = np.clip(self.eps * self.decay_factor, 0.01, 1.0)

    def step(self, sample: Sample) -> None:
        """Update the agent's knowledge and policy correspondingly"""
        state, action, reward, next_state = sample
        next_action = self.forward(next_state)

        coordinates = self.Q.get_coordinates(state)
        next_coordinates = self.Q.get_coordinates(next_state)

        phi = np.sum([self.phi[coordinate] for coordinate in coordinates], axis=0)
        next_phi = np.sum([self.phi[coordinate] for coordinate in next_coordinates], axis=0)

        # print(phi.shape)
        # print(coordinates, phi.sum())

        error = reward + self.gamma * next_phi.dot(self.W) - phi.dot(self.W)

        self.W = self.W + self.lr * error * phi

        self.Q.update(state, action, phi.dot(self.W))


if __name__ == '__main__':
    positions_range = (-1.2, 0.6)
    velocity_range = (-0.07, 0.071)
    low, high = (positions_range[0], velocity_range[0]), (positions_range[1], velocity_range[1])

    tiling_specs = [
        ((10, 10), (0.0, 0.0)),
        ((10, 10), (0.04, 0.03)),
    ]

    agent = QLearning([low, high], num_actions=3, tiling_specs=tiling_specs, feat_dim=100, eps=0.2, decay_factor=0.99, lr=0.2, gamma=0.9)

    for _ in range(5):
        print(agent((-1.0, 0.0)), end=' ')
        print(agent.Q.get((-1.0, 0.0), 0), end=' | ')
        reward = np.random.randint(1, 1000)
        agent.step(Sample((-1.0, 0.0), 0, reward, (0.0, 0.0)))
        print(agent((-1.0, 0.0)), end=' ')
        print(agent.Q.get((-1.0, 0.0), 0))
