import imp
from typing import Tuple
import numpy as np
from pprint import pprint
from numpy.typing import ArrayLike

from data_loader.environments import GridWorldEnvironment, Directions
from models.value_iteration_agent import ValueIterationAgent
from models.policy_iteration_agent import PolicyIterationAgent
from utils.utils import numpy_to_gif, dict_to_gif

np.set_printoptions(linewidth=np.inf)


def get_direction(state: int, action: int) -> Tuple[int, str]:
    """Get the actual state and the direction of the action."""
    if state >= 9:
        state += 1
    if action == Directions.UP:
        return state, '^'
    elif action == Directions.DOWN:
        return state, 'v'
    elif action == Directions.LEFT:
        return state, '<'
    elif action == Directions.RIGHT:
        return state, '>'
    else:
        return state, '-'


if __name__ == '__main__':

    env = GridWorldEnvironment(gamma=0.9)
    agent = ValueIterationAgent(env.observation_space.n)
    # agent = PolicyIterationAgent(env.observation_space.n)

    info = agent.learn(env)
    # --------------------< For Policy Iteration Agent >--------------------
    # policies = [dict(get_direction(state, action) for state, action in policy.items()) for policy in info]
    # dict_to_gif(policies, 'artifacts/policy_iteration_agent_gif.gif')
    # --------------------</ For Policy Iteration Agent >--------------------


    # --------------------< For Value Iteration Agent >--------------------
    value_functions_array = info['val_func']
    value_func_diff_array = info['diff']

    numpy_to_gif(value_functions_array, 'artifacts/value_functions.gif', title='Value functions')
    numpy_to_gif(value_func_diff_array, 'artifacts/value_functions_diff.gif', title='Value functions diff')
    # --------------------</ For Value Iteration Agent >--------------------
