import numpy as np
import hydra
import pathlib
from tqdm import tqdm

from data_loader.environments import MultiArmBanditEnvironment
from data_loader.bandit_arm_reward_initializer import BanditArmRewardInitializer
from models.epsilon_greedy import EpsilonGreedyAgent



@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    # np.random.seed(cfg.seed)
    env = MultiArmBanditEnvironment(arm_initializer=BanditArmRewardInitializer('binomial'))
    agent = EpsilonGreedyAgent(0.3, env.num_arms) #, initial_temp=1.0, decay_factor=0.99)

    obs = env.reset()
    for _ in range(5000):
        action = agent(obs)
        obs, reward, done, info = env.step(action)
        agent.update_mean(action, reward)
        print(f"obs: {obs}, reward: {reward}, done: {done}, info: {info}")


if __name__ == '__main__':
    # pathlib.Path(f'{pathlib.Path.cwd()}/figs/').mkdir(parents=True, exist_ok=True)
    main()