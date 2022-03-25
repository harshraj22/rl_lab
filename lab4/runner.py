import sys
import random
import logging
from tqdm import tqdm

import gym

from data_loader.environments import LinearEnv
from models.on_policy_mc import FirstVisitMonteCarlo
from models.q_learning import QLearning
from models.sarsa import SARSA
from utils.utils import Sample
from utils.wrappers import LinearEnvWrapper
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
    for episode in tqdm(range(config.num_episodes)):
        state = env.reset()
        done, reward = False, 0
        trajectory = []

        while not done:
            action = agent.forward(state)
            # logger.info(f'State: {state} | Action: {action}')
            next_state, reward, done, _ = env.step(action)
            trajectory.append(Sample(state, action, reward, next_state))
            state = next_state
            if agent.mode == 'online':
                agent.step(Sample(state, action, reward, next_state))
            # env.render()
        
        if agent.mode == 'offline':
            # calculate the discounted sum of returns
            inverted_returns = [0]
            for sample in reversed(trajectory):
                inverted_returns.append(sample.reward + inverted_returns[-1])

            returns = list(reversed(inverted_returns[1:]))
            visited = set()
            for sample, return_ in zip(trajectory, returns):
                if (sample.state, sample.action) not in visited:
                    agent.step(Sample(sample.state, sample.action, return_, sample.next_state))
                visited.add((sample.state, sample.action))
        else:
            returns = [min(1, reward)]
            
        agent.learn()
        recieved_perfect += int(returns[0])
        logger.info(f'Episode {episode}/{config.num_episodes}, Reward: {returns[0]:.3f}, Epsilon: {agent.eps:.3f}, time: {len(trajectory)} | recieved perfect: {recieved_perfect}')
        logger.info(f'Q: {agent.Q}')
        # logger.info(f'Returns: {returns}\n Trajectory: {trajectory}')
    return agent


if __name__ == '__main__':
    config = OmegaConf.load('conf/config.yaml')
    random.seed(config.seed)

    # ToDo: Create agent, environment depending on config
    # env = gym.make('FrozenLake-v1') # A discrete Action Space environment
    env = LinearEnvWrapper(LinearEnv(max_time=8))
    env.seed(config.seed)
    # agent = FirstVisitMonteCarlo(
    #     env.observation_space.n,
    #     env.action_space.n,
    #     decay_factor=config.agent.montecarlo.decay_factor,
    #     eps=config.agent.montecarlo.eps
    #     )
    # agent = SARSA(
    #     env.observation_space.n,
    #     env.action_space.n,
    #     eps=config.agent.sarsa.eps,
    #     decay_factor=config.agent.sarsa.decay_factor,
    #     lr=config.agent.sarsa.lr,
    #     gamma=config.agent.sarsa.gamma
    #     )
    agent = QLearning(
        env.observation_space.n,
        env.action_space.n,
        eps=config.agent.qlearning.eps,
        decay_factor=config.agent.qlearning.decay_factor,
        lr=config.agent.qlearning.lr,
        gamma=config.agent.qlearning.gamma
        )

    # mentioned in algo, to fill Q[terminal_state][*] with 0
    # agent.Q[-1] = 0
    print(f'Details about env: Actions: {env.action_space.n} | States: {env.observation_space.n}')
    trained_agent = learn(agent, env, config)

    # Test the agent
    # done, state, reward = False, env.reset(), 0
    # while not done:
    #     action = trained_agent.forward(state)
    #     state, reward, done, _ = env.step(action)
    #     env.render()
    # print(f'Reward: {reward}')

    
        