import numpy as np
import hydra
import pathlib
from tqdm import tqdm

from data_loader.environments import MultiArmBanditEnvironment
from data_loader.bandit_arm_reward_initializer import BanditArmRewardInitializer



@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    # np.random.seed(cfg.seed)
    env = MultiArmBanditEnvironment(arm_initializer=BanditArmRewardInitializer('binomial'))

    obs = env.reset()
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"obs: {obs}, reward: {reward}, done: {done}, info: {info}")


if __name__ == '__main__':
    # pathlib.Path(f'{pathlib.Path.cwd()}/figs/').mkdir(parents=True, exist_ok=True)
    main()