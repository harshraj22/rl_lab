import sys
from typing import Callable
import numpy as np
import gym
import logging
from tqdm import tqdm

sys.path.insert(0, '../')
from base.iteration_agent import IterationAgent
from utils.utils import PreserveEnvStateManager
from data_loader.environments import GridWorldEnvironment


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(name)s] [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


class ValueIterationAgent(IterationAgent):
    """A generic class for a value iteration agent."""
    def __init__(self, num_states: int, num_actions: int = 4) -> None:
        super(ValueIterationAgent, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.value_functions = np.zeros(self.num_states)
        self.actions = 0

    def learn(self, env: GridWorldEnvironment, num_timesteps: int = 100) -> None:
        """Learn from the environment."""
        for current_timestep in tqdm(range(num_timesteps), f'Learning for {num_timesteps}', total=num_timesteps):
            new_values = np.full(self.value_functions.shape, -np.inf)
            new_action = np.zeros(self.value_functions.shape)
            for state in range(self.num_states):
                env.state = state
                _best_action = np.zeros(self.num_actions)
                for action in range(self.num_actions):
                    # Taking the action 'num_stpes' times, to procure the estimated
                    # value of R(i, a, j) + gamma * V(j).
                    _current_sum, num_steps = 0, 100
                    for _ in range(num_steps):
                        with PreserveEnvStateManager(env) as cur_env:
                            _, reward, _, _ = cur_env.step(action)
                            _current_sum += reward + cur_env.gamma * self.value_functions[cur_env.state]
                    _best_action[action] = _current_sum / num_steps
                    new_values[state] = max(new_values[state], _current_sum / num_steps)
                new_action[state] = np.argmax(_best_action)
            
            self.value_functions = new_values
            self.actions = new_action
            # print the value function

    def action(self, state: int) -> int:
        """Select an action."""
        return self.actions