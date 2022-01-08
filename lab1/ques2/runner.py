import numpy as np
import hydra
import matplotlib.pyplot as plt
import pathlib


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    # np.random.seed(cfg.seed)
    pass

if __name__ == '__main__':
    pathlib.Path(f'{pathlib.Path.cwd()}/figs/').mkdir(parents=True, exist_ok=True)
    main()