import numpy as np

from data_loader.environments import GridWorldEnvironment
from models.value_iteration_agent import ValueIterationAgent

np.set_printoptions(linewidth=np.inf)

env = GridWorldEnvironment(gamma=0.9)
agent = ValueIterationAgent(env.observation_space.n)
print('Initial: ', agent.value_functions)

agent.learn(env)
print('Final: ', agent.value_functions)