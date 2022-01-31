import numpy as np
import hydra
import pathlib
from tqdm import tqdm
import wandb
import logging
import sys

from data_loader.environments import MultiArmBanditEnvironment
from data_loader.bandit_arm_reward_initializer import BanditArmRewardInitializer
from models import ReinforceAgent, EpsilonGreedyAgent, UCBAgent, ThompsonSamplingAgent, SoftmaxAgent


wandb_run = wandb.init(project="multi_arm_bandit", entity="harshraj22", mode="disabled")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(name)s] [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    np.random.seed(cfg.seed)

    env = MultiArmBanditEnvironment(
        arm_initializer=BanditArmRewardInitializer('gaussian'),
        num_arms=cfg.env.num_arms,
        total_timesteps=cfg.total_timesteps
        )
    # agent = EpsilonGreedyAgent(0.3, env.num_arms, initial_temp=1.0, decay_factor=1.001)
    # agent = SoftmaxAgent(env.num_arms)
    # agent = UCBAgent(env.num_arms)
    agent = ThompsonSamplingAgent(env.num_arms, underlying_dist='gaussian')
    # agent = ReinforceAgent(env.num_arms, baseline=True)

    wandb_run.name = str(agent)

    obs = env.reset()
    for chance in tqdm(range(1, cfg.total_timesteps + 1)):
        action = agent(obs)
        obs, reward, done, info = env.step(action)
        agent.update_mean(action, reward)
        wandb.log({
            "optimal_arm_percentage": info['optimal_arm_hits'] / chance
        })
        # logger.info(f"obs: {obs}, reward: {reward}, done: {done}, info: {info}, agent: {agent.__class__.__name__}")
    logger.info(f"info: {info}")

if __name__ == '__main__':
    # pathlib.Path(f'{pathlib.Path.cwd()}/figs/').mkdir(parents=True, exist_ok=True)
    main()
    # print(wandb.)
    # print(logger.handlers)