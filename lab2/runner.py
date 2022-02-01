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
    wandb_run = wandb.init(project="multi_arm_bandit", entity="harshraj22", mode=cfg.wandb_tracking)
    np.random.seed(cfg.seed)

    env = MultiArmBanditEnvironment(
        arm_initializer=BanditArmRewardInitializer(cfg.env.reward_dist),
        num_arms=cfg.env.num_arms,
        total_timesteps=cfg.total_timesteps
        )

    if cfg.agent.type == 'eps_greedy':
        agent = EpsilonGreedyAgent(cfg.agent.eps_greedy.epsilon, env.num_arms, initial_temp=cfg.agent.eps_greedy.initial_temp, decay_factor=cfg.agent.eps_greedy.decay_factor)
    elif cfg.agent.type == 'softmax':
        agent = SoftmaxAgent(env.num_arms)
    elif cfg.agent.type == 'ucb':
        agent = UCBAgent(env.num_arms)
    elif cfg.agent.type == 'thompson_sampling':
        agent = ThompsonSamplingAgent(env.num_arms, underlying_dist=cfg.env.reward_dist)
    elif cfg.agent.type == 'reinforce':
        agent = ReinforceAgent(env.num_arms, baseline=cfg.agent.reinforce.baseline, beta=cfg.agent.reinforce.beta, alpha=cfg.agent.reinforce.alpha)
    else:
        raise ValueError(f'Unknown agent type: {cfg.agent.type}. Please choose from: eps_greedy, softmax, ucb, thompson_sampling, reinforce')

    wandb_run.name = str(agent)

    obs = env.reset()
    for chance in tqdm(range(1, cfg.total_timesteps + 1)):
        action = agent(obs)
        obs, reward, done, info = env.step(action)
        # logger.info(f'Env: {env.reward_distributions} | Action: {action} | Reward: {reward:.3f}')
        agent.update_mean(action, reward)
        wandb.log({
            f"optimal_arm_percentage: {cfg.env.num_arms} arms, {cfg.env.reward_dist} distribution": info['optimal_arm_hits'] / chance
        })
        # logger.info(f"obs: {obs}, reward: {reward}, done: {done}, info: {info}, agent: {agent.__class__.__name__}")
    logger.info(f"info: {info}")

if __name__ == '__main__':
    # pathlib.Path(f'{pathlib.Path.cwd()}/figs/').mkdir(parents=True, exist_ok=True)
    main()
    # print(wandb.)
    # print(logger.handlers)

    # python3 runner.py -m agent.type=reinforce env.reward_dist=gaussian seed=32,12,23,43,54,65,76,234,2342,3423,42342,34,234,234,234,23 wandb_tracking=disabled agent.reinforce.baseline=False total_timesteps=5000 env.num_arms=5