import gym

from models.on_policy_mc import FirstVisitMonteCarlo
from utils.utils import Sample
from omegaconf import OmegaConf

""" General way for offline agents:

    1. Play an episode, and record the `utils.utils.Sample` at each timestep
    2. Calculate the discounted sum of returns at each time step
    3. Call `agent.step(...)` at each timestep
    4. Call `agent.learn()` at the end of the episode
"""


if __name__ == '__main__':
    config = OmegaConf.load('conf/config.yaml')

    # ToDo: Create agent, environment depending on config
    env = gym.make('Taxi-v3') # A discrete Action Space environment
    agent = FirstVisitMonteCarlo(
        env.observation_space.n,
        env.action_space.n,
        decay_factor=config.agent.montecarlo.decay_factor,
        eps=config.agent.montecarlo.eps
        )

    print(f'Details about env: Actions: {env.action_space.n} | States: {env.observation_space.n}')

    for episode in range(config.num_episodes): 
        state = env.reset()
        done = False
        trajectory = []

        while not done:
            action = agent.forward(state)
            next_state, reward, done, _ = env.step(action)
            trajectory.append(Sample(state, action, reward, next_state))
            state = next_state
        
        # calculate the discounted sum of returns
        returns = [0]
        for sample in reversed(trajectory):
            returns.append(sample.reward + returns[-1])

        returns = returns[:0:-1]
        for sample, return_ in zip(trajectory, returns):
            agent.step(Sample(sample.state, sample.action, return_, sample.next_state))
        
        agent.learn()
        print(f'Episode {episode}/{config.num_episodes}, Reward: {returns[0]}, Epsilon: {agent.eps:.3f}')
        