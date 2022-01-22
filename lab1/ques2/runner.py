import numpy as np
import hydra
import matplotlib.pyplot as plt
import pathlib
import sys
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler(sys.stdout))


def generate_data_point(sample_size: int = 10) -> int:
    samples = np.random.uniform(0, 1, sample_size)
    return samples.mean()

def generate_using_CLT(num_samples: int, mean: float, variance: float):
    # np.random.seed(cfg.seed)
    data_points = [generate_data_point(100) for _ in range(num_samples)]
    data_points = np.array(data_points)

    data_points = (data_points - data_points.mean()) / data_points.std()

    std = np.sqrt(variance)
    data_points = data_points * std + mean

    return data_points


def generate_using_Box_Muller_Transform(num_samples: int, mean: float, variance: float):
    data_points = []
    std = np.sqrt(variance)
    for _ in range(num_samples // 2):
        u1, u2 = np.random.uniform(0, 1), np.random.uniform(0, 1)
        x1 = np.cos(2 * np.pi * u2) * np.sqrt(-2 * np.log(u1))
        x2 = np.sin(2 * np.pi * u2) * np.sqrt(-2 * np.log(u1))
        x1 = x1 * std + mean
        x2 = x2 * std + mean
        data_points.extend([x1, x2])

    return data_points


@hydra.main(config_path="conf", config_name="configs")
def main(cfg):
    data_points = generate_using_CLT(cfg.num_samples, cfg.mean, cfg.variance)

    logger.info(f'Mean: {data_points.mean():.3f}, Std: {data_points.std():.3f}')

    plt.hist(data_points, bins=100, density=True)
    plt.title(f"Histogram using {cfg.num_samples} samples")
    plt.savefig(f'{hydra.utils.get_original_cwd()}/figs/ques2.png')
    

if __name__ == '__main__':
    pathlib.Path(f'{pathlib.Path.cwd()}/figs/').mkdir(parents=True, exist_ok=True)
    main()