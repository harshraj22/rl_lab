import numpy as np
import hydra
import matplotlib.pyplot as plt
import pathlib


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    # np.random.seed(cfg.seed)

    if cfg.dist_type == "multinomial":
        numbers = np.random.multinomial(n=1, size=(cfg.num_pts,), pvals=[0.2, 0.4, 0.3, 0.1]).argmax(axis=1)
        range = (0, 4)
    elif cfg.dist_type == "uniform":
        numbers = np.random.uniform(low=0, high=1, size=(cfg.num_pts,)) #.argmax(axis=1)
        range = (0, 1)
    elif cfg.dist_type == "gaussian":
        numbers = np.random.normal(loc=0, scale=1, size=(cfg.num_pts,)) #.argmax(axis=1)
        range = (numbers.min(), numbers.max())
    elif cfg.dist_type == "exponential":
        numbers = np.random.exponential(scale=0.5, size=(cfg.num_pts,))
        range = (numbers.min(), numbers.max())
    else:
        raise ValueError("dist_type must be one of 'multinomial', 'uniform', 'gaussian', 'exponential'")

    # ToDo: update the num of bins depending on the range of the distribution and change in len(numbers)
    plt.hist(numbers, bins=20, range=range, density=True)
    plt.xlabel('Random Variable')
    plt.ylabel('Fraction of num of hits')
    plt.title(f'Sampling from {cfg.dist_type} Distribution: {cfg.num_pts} samples')
    # print(numbers)

    plt.savefig(pathlib.Path(f"{hydra.utils.get_original_cwd()}/figs/ques1_{cfg.dist_type}_{cfg.num_pts}.png"))

if __name__ == '__main__':
    pathlib.Path(f'{pathlib.Path.cwd()}/figs/').mkdir(parents=True, exist_ok=True)
    main()