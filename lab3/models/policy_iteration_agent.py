import sys
from typing import Callable, Dict, List
import numpy as np
import gym
import logging
from tqdm import tqdm
import wandb
from copy import deepcopy

sys.path.insert(0, '../')
from base.iteration_agent import IterationAgent
from utils.utils import PreserveEnvStateManager
from data_loader.environments import GridWorldEnvironment, Directions


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(name)s] [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


class PolicyIterationAgent(IterationAgent):
    """A generic class for a value iteration agent."""
    def __init__(self, num_states: int, num_actions: int = 4) -> None:
        wandb_run = wandb.init(project="value-policy-iteration", entity="harshraj22", mode='disabled')
        super(PolicyIterationAgent, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.value_functions = np.zeros(self.num_states)
        self.policy = {state: 0 for state in range(self.num_states)}
        # self.policy = {
        #     0: Directions.RIGHT,
        #     1: Directions.RIGHT,
        #     2: Directions.RIGHT,
        #     3: Directions.UP,
        #     4: Directions.RIGHT,
        #     5: Directions.UP,
        #     6: Directions.LEFT,
        #     7: Directions.UP,
        #     8: Directions.UP,
        #     9: Directions.UP,
        #     10: Directions.DOWN,
        #     11: Directions.UP,
        #     12: Directions.LEFT,
        #     13: Directions.UP,
        #     14: Directions.LEFT
        # }

    def learn(self, env: GridWorldEnvironment, num_timesteps: int = 10, num_episode: int = 100) -> List[Dict]:
        """Learn from the environment."""
        policies = []
        for _ in tqdm(range(num_episode), f'Learning for {num_episode}', total=num_episode):
            # 1. policy evaluation
            self.policy_evaluation(env, num_timesteps)

            # 2. policy improvement
            # update the policy, using argmax over value functions
            for state in range(self.num_states):
                _n_times = 100
                _best_action = np.zeros(self.num_actions)
                for _current_action in range(self.num_actions):
                    # argmax over a (Q_mu(i, a))
                    for _ in range(_n_times):
                        with PreserveEnvStateManager(env) as cur_env:
                            cur_env.state = state
                            next_state, reward, _, _ = cur_env.step(_current_action)
                            _best_action[_current_action] += cur_env.gamma * self.value_functions[next_state] + reward
                            # assert reward >= 0, f'Negative aaya'
                            
                self.policy[state] = np.argmax(_best_action / _n_times)
                # logger.info(f'state: {state}, Best action: {_best_action}')
                    # self.policy[state] = np.argmax([self.value_functions[cur_env.step(action)[0]] for action in range(self.num_actions)])

            policies.append(deepcopy(self.policy))
        return policies

    def policy_evaluation(self, env: GridWorldEnvironment, num_timesteps: int = 1000) -> None:
        """Evaluate the policy"""
        value_functions_array, value_func_diff_array = [np.zeros_like(self.value_functions)], [np.zeros_like(self.value_functions)]

        for current_timestep in tqdm(range(num_timesteps), f'Learning value fun for {num_timesteps}', total=num_timesteps):
            new_values = np.full(self.value_functions.shape, -np.inf)
            for state in range(self.num_states):
                env.state = state
                action = self.action(state)
                # Taking the action 'num_stpes' times, to procure the estimated
                # value of R(i, a, j) + gamma * V(j).
                _current_sum, num_steps = 0, 100
                for _ in range(num_steps):
                    with PreserveEnvStateManager(env) as cur_env:
                        _, reward, _, _ = cur_env.step(action)
                        # assert reward >= 0, f'Current state: {state}, action: {action}, reward: {reward}'
                        _current_sum += reward + cur_env.gamma * self.value_functions[cur_env.state]
                new_values[state] = _current_sum / num_steps

            value_func_diff = np.abs(self.value_functions - new_values)
            wandb.log({
                'avg_value_func_diff_Policy_iter': np.mean(value_func_diff),
                'max_value_func_diff_Policy_iter': np.max(value_func_diff),
            })
            
            self.value_functions = new_values

            value_functions_array.append(self.value_functions)
            value_func_diff_array.append(value_func_diff)
        
        # new_values = list(new_values)
        # new_values.insert(9, 0)
        # tqdm.write(f'Value functions: \n{np.array(new_values).reshape((4, 4))}')

    def action(self, state: int) -> int:
        """Select an action."""
        return self.policy[state]
        