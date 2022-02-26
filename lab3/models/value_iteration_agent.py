import sys
import numpy as np
import gym
import logging


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

    def learn(self, env: GridWorldEnvironment, num_timesteps: int = 1000) -> None:
        """Learn from the environment."""
        for current_timestep in range(num_timesteps):
            new_values = np.full(self.value_functions.shape, -np.inf)
            for state in range(self.num_states):
                env.state = state
                for action in range(self.num_actions):
                    with PreserveEnvStateManager(env) as cur_env:
                        # logger.info(f'State: {cur_env.state}, Action: {action}')
                        _, reward, _, _ = cur_env.step(action)
                        # logger.info(f'New state: {cur_env.state}, Reward: {reward}')
                        new_values[state] = max(new_values[state], reward + cur_env.gamma * self.value_functions[cur_env.state])
                # logger.info(f'State: {state}, Reward: {reward}')

            self.value_functions = new_values


    def action(self, state: int) -> int:
        """Select an action."""
        pass