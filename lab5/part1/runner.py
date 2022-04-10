import sys
import random
import logging
from tqdm import tqdm
import numpy as np
from math import ceil

import gym
import wandb

from models.q_learning import QLearning
from models.sarsa import SARSA
from utils.utils import Sample
from utils.wrappers import LinearEnvWrapper, MountainCarEnvWrapper
from base.baseagent import BaseAgent
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(name)s] [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


""" General way for offline agents:

    1. Play an episode, and record the `utils.utils.Sample` at each timestep
    2. Calculate the discounted sum of returns at each time step
    3. Call `agent.step(...)` at each timestep
    4. Call `agent.learn()` at the end of the episode
"""


def learn(agent: BaseAgent, env: gym.Env, config) -> BaseAgent:
    """Function to train the agents that learn offline.
    Parameters:
    ----------
    agent : BaseAgent
        The agent to train.
    env : gym.Env
        The environment to train the agent on.
    config : dict
        The configuration for the agent and environment and experiemnt.
    
    Returns:
    -------
    agent : BaseAgent
        The trained agent.
    """
    recieved_perfect = 0
    for episode in tqdm(range(config.num_episodes), desc="Episode"):
        state = env.reset()
        done, reward = False, 0
        trajectory = []

        while not done:
            action = agent(state)
            # logger.info(f'State: {state} | Action: {action}')
            next_state, reward, done, info = env.step(action)
            trajectory.append(Sample(state, action, reward, next_state))
            # state = next_state
            if agent.mode == 'online':
                agent.step(Sample(state, action, reward, next_state))
            state = next_state
            # env.render('human')
        
        # calculate the discounted sum of returns
        inverted_returns = [0]
        for sample in reversed(trajectory):
            inverted_returns.append(sample.reward + inverted_returns[-1])

        returns = list(reversed(inverted_returns[1:]))
        wandb.log({
            f'{config.env}/Reward': returns[0],
            f'{config.env}/eps': agent.eps
        })
        if agent.mode == 'offline':
            visited = set()
            for sample, return_ in zip(trajectory, returns):
                if (sample.state, sample.action) not in visited:
                    agent.step(Sample(sample.state, sample.action, return_, sample.next_state))
                visited.add((sample.state, sample.action))
        else:
            # returns = [reward]
            pass
            
        agent.learn()
        recieved_perfect += ceil(np.clip(reward, 0, 1))
        logger.info(f'Episode {episode}/{config.num_episodes}, Reward: {returns[0]:.3f}, Epsilon: {agent.eps:.3f}, time: {len(trajectory)} | recieved perfect: {recieved_perfect}')
        # logger.info(f'Q: {agent.Q}')
        # logger.info(f'Returns: {returns}')
    return agent


if __name__ == '__main__':
    # Overwrite the default configuration with the one from the command line
    config = OmegaConf.load('conf/config.yaml')
    config = OmegaConf.merge(config, OmegaConf.from_cli())
    random.seed(config.seed)
    np.random.seed(config.seed)

    wandb.init(project='SARSA-MCMC-QLearning', mode='disabled')
    wandb.run.name = config.agent.type

    # env = gym.make('FrozenLake-v1') # A discrete Action Space environment
    # env = LinearEnvWrapper(LinearEnv(max_time=8))
    env = gym.make(config.env)

    # ToDo: Automate the following
    positions_range = (-1.2, 0.6)
    velocity_range = (-0.07, 0.071)
    low, high = (positions_range[0], velocity_range[0]), (positions_range[1], velocity_range[1])

    tiling_specs = [
        ((10, 10), (0.0, 0.0)),
        ((10, 10), (0.04, 0.03)),
    ]

    agent = QLearning([low, high], num_actions=3, tiling_specs=tiling_specs, feat_dim=100, eps=0.2, decay_factor=0.99, lr=0.2, gamma=0.9)

    learn(agent, env, config)
    

    # mentioned in algo, to fill Q[terminal_state][*] with 0
    # # agent.Q[-1] = 0
    # print(f'Details about env: Actions: {env.action_space.n} | States: {env.observation_space.n}')
    # trained_agent = learn(agent, env, config)



    # Test the agent
    # done, state, reward = False, env.reset(), 0
    # while not done:
    #     action = trained_agent.forward(state)
    #     state, reward, done, _ = env.step(action)
    #     env.render()
    # print(f'Reward: {reward}')

    
        