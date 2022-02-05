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
        agent = SoftmaxAgent(env.num_arms, initial_temp=cfg.agent.softmax.initial_temp, decay_factor=cfg.agent.softmax.decay_factor)
    elif cfg.agent.type == 'ucb':
        agent = UCBAgent(env.num_arms)
    elif cfg.agent.type == 'thompson_sampling':
        agent = ThompsonSamplingAgent(env.num_arms, underlying_dist=cfg.env.reward_dist)
    elif cfg.agent.type == 'reinforce':
        agent = ReinforceAgent(env.num_arms, baseline=cfg.agent.reinforce.baseline, beta=cfg.agent.reinforce.beta, alpha=cfg.agent.reinforce.alpha)
    else:
        raise ValueError(f'Unknown agent type: {cfg.agent.type}. Please choose from: eps_greedy, softmax, ucb, thompson_sampling, reinforce')

    wandb_run.name = str(agent)

    rewards = np.zeros((cfg.num_runs, cfg.total_timesteps))
    optimal_arm_hits = np.zeros((cfg.num_runs, cfg.total_timesteps))
    mu_star = env.optimal_mean

    for run_index in tqdm(range(cfg.num_runs)):
        agent.reset()
        obs = env.reset()

        for current_timestep in tqdm(range(1, cfg.total_timesteps + 1)):
            action = agent(obs)
            obs, reward, done, info = env.step(action)
            rewards[run_index][current_timestep - 1] = reward
            optimal_arm_hits[run_index][current_timestep - 1] = info['optimal_arm_hits'] / current_timestep
            # logger.info(f'Env: {env.reward_distributions} | Action: {action} | Reward: {reward:.3f}')
            agent.update_mean(action, reward)
        logger.info(f"info: {info}")

    mean_rewards = np.mean(rewards, axis=0)
    mean_optimal_arm_hits = np.mean(optimal_arm_hits, axis=0)
    cummulative_mean_reward = 0
    for mean_reward, mean_optimal_arm_hit, current_timestep in zip(mean_rewards, mean_optimal_arm_hits, range(cfg.total_timesteps)):
        cummulative_mean_reward += mean_reward
        regret = mu_star * current_timestep - cummulative_mean_reward
        wandb.log({
            f"mean_reward: {cfg.env.num_arms} arms, {cfg.env.reward_dist} distribution": mean_reward,
            f"optimal_arm_percentage: {cfg.env.num_arms} arms, {cfg.env.reward_dist} distribution": mean_optimal_arm_hit,
            f"regret: {cfg.env.num_arms} arms, {cfg.env.reward_dist} distribution": regret
        })

    # logger.info(f'\nRewards: \n{rewards} \nMean: {mean_rewards}')
    # logger.info(f'\nOptimal arm hits: \n{optimal_arm_hits} \nMean: {mean_optimal_arm_hits}')
    # logger.info(f'Mean reward: {mean_rewards.shape}')
    # logger.info(f'Env: {env}')

if __name__ == '__main__':
    # pathlib.Path(f'{pathlib.Path.cwd()}/figs/').mkdir(parents=True, exist_ok=True)
    main()
    # print(wandb.)
    # print(logger.handlers)

    # python3 runner.py -m agent.type=reinforce env.reward_dist=gaussian seed=32,12,23,43,54,65,76,234,2342,3423,42342,34,234,234,234,23 wandb_tracking=disabled agent.reinforce.baseline=False total_timesteps=5000 env.num_arms=5