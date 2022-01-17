import numpy as np
import matplotlib.pyplot as plt
import pathlib
from utils.utils import sqrt_sin, calculate_area


def main():
    # ------- Part A -------
    x = np.linspace(0, np.pi, 60)
    sin_x = np.sin(x)
    sqrt_sin_x = np.sqrt(sin_x)

    plt.plot(x, sqrt_sin_x)
    plt.title('x vs √sin(x)')
    plt.xlabel('x')
    plt.ylabel('√sin(x)')
    plt.savefig('figs/a.png')
    plt.clf()


    # ------- Part B -------
    x = np.linspace(0, np.pi, 60)
    sin_x = np.sin(x)
    sqrt_sin_x = np.sqrt(sin_x)
    exp_minus_x2 = -np.exp(x**2)

    plt.plot(x, sqrt_sin_x * exp_minus_x2)
    plt.title('x vs √sin(x)exp(-x2)')
    plt.xlabel('x')
    plt.ylabel('√sin(x)exp(-x2)')
    plt.savefig('figs/b.png')
    plt.clf()    


if __name__ == "__main__":
    pathlib.Path(f'{pathlib.Path.cwd()}/figs/').mkdir(parents=True, exist_ok=True)
    # main()
    print('Approx Area: ', calculate_area(lambda x: 1, num_bins=2))