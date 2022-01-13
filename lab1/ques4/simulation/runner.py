import numpy as np
import hydra
import pathlib
from data_loader.environments import SnakeAndLadderEnv
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


def play_a_game(max_timesteps: int = 40) -> int:
    """Play a single game of Snake and Ladder.

    Args:
        max_timesteps (int, optional): Number of timesteps for which a single game
          lasts. Defaults to 40.

    Returns:
        int: The reward of the game.
    """
    env = SnakeAndLadderEnv(view=(3, 3))
    obs, done = env.reset(), False
    reward = 0

    # play a game
    while not done:
        action = np.random.randint(1, 7)
        logger.debug(f'Action taken: {action}')
        obs, reward, done, _ = env.step(action)
        # env.render()
    logger.info(f'Reward: {reward}')
    env.close()
    return reward


def simulate(num_games: int, max_timestep_per_game: int) -> None:
    """Simulate the experiment of playing a series of games, and finding the
    probability of winning using the rewards returned.

    Args:
        num_games (int): Number of games to simulate.
        max_timestep_per_game (int): Maximum number of timesteps per game. This 
            is used to exit the game with 0 reward in case the player ends up in
            a dead state.
    """
    rewards = []
    for _ in tqdm(range(num_games), desc='Simulating games'):
        rewards.append(play_a_game(max_timesteps=max_timestep_per_game))
    logger.debug(rewards)
    print(f'Probability of winning: {np.mean(rewards)}')


@hydra.main(config_path="conf", config_name="configs")
def main(cfg):
    pathlib.Path(f'{pathlib.Path.cwd()}/{cfg.exp.data_dir}/').mkdir(parents=True, exist_ok=True)
    simulate(num_games=cfg.exp.num_games, max_timestep_per_game=cfg.exp.max_timestep_per_game)


if __name__ == '__main__':
    main()