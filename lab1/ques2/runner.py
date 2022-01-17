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


@hydra.main(config_path="conf", config_name="configs")
def main(cfg):
    np.random.seed(cfg.seed)
    data_points = [generate_data_point(100) for _ in range(cfg.num_samples)]
    data_points = np.array(data_points)

    data_points = (data_points - data_points.mean()) / data_points.std()

    std = np.sqrt(cfg.variance)
    data_points = data_points * std + cfg.mean

    logger.info(f'Mean: {data_points.mean():.3f}, Std: {data_points.std():.3f}')

    plt.hist(data_points, bins=100, density=True)
    plt.title(f"Histogram using {cfg.num_samples} samples")
    plt.savefig(f'{hydra.utils.get_original_cwd()}/figs/ques2.png')
    

if __name__ == '__main__':
    pathlib.Path(f'{pathlib.Path.cwd()}/figs/').mkdir(parents=True, exist_ok=True)
    main()