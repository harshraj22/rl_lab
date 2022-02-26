import sys
import numpy as np
import gym

sys.path.insert(0, '../')
from base.iteration_agent import IterationAgent
from utils.utils import PreserveEnvStateManager


class ValueIterationAgent(IterationAgent):
    """A generic class for a value iteration agent."""
    def __init__(self, num_states: int, num_actions: int = 4) -> None:
        super(ValueIterationAgent, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.values = np.zeros(self.num_states)

    def learn(self, env: gym.Env, num_timesteps: int = 1000) -> None:
        """Learn from the environment."""
        for current_timestep in range(num_timesteps):
            new_values = np.zeros_like(self.values)
            for state in range(self.num_states):
                for action in range(self.num_actions):
                    with PreserveEnvStateManager(env) as cur_env:
                        _, reward, _, _ = cur_env.step(action)
                        # TODO: Add/think about the discounting factor
                        new_values[state] = max(new_values[state], reward + self.values[state])

            self.values = new_values


    def action(self, state: int) -> int:
        """Select an action."""
        pass