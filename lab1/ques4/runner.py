import numpy as np
import hydra
import pathlib
from data_loader.environments import SnakeAndLadderEnv


@hydra.main(config_path="conf", config_name="configs")
def main(cfg):
    # np.random.seed(cfg.seed)
    env = SnakeAndLadderEnv(view=(3, 3))
    obs, done = env.reset(), False
    rewards = []

    # play a game
    while not done:
        action = np.random.randint(1, 3)
        print(f'Action taken: {action}')
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        env.render()
    print(f'Reward: {np.sum(rewards)}')
    env.close()


if __name__ == '__main__':
    pathlib.Path(f'{pathlib.Path.cwd()}/figs/').mkdir(parents=True, exist_ok=True)
    main()