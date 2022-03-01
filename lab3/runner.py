import imp
import numpy as np
from pprint import pprint

from data_loader.environments import GridWorldEnvironment
from models.value_iteration_agent import ValueIterationAgent
from models.policy_iteration_agent import PolicyIterationAgent

np.set_printoptions(linewidth=np.inf)

env = GridWorldEnvironment(gamma=0.9)
agent = ValueIterationAgent(env.observation_space.n)
print('Initial: ', agent.value_functions)

agent.learn(env)
print('Final: ', agent.value_functions)

pprint(agent.actions)