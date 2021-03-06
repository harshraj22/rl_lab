import numpy as np
import matplotlib.pyplot as plt
import pathlib
from utils.utils import sqrt_sin, calculate_area, sqrt_sin_exp_minus_x2


def print_areas(num_bins: int = 20_000) -> None:
    print(f'Calculated Area of √sin(x) over [0, pi] is: {calculate_area(sqrt_sin, num_bins)}')
    print(f'Calculated Area of √sin(x)exp(-x2) over [0, pi] is: {calculate_area(sqrt_sin_exp_minus_x2, num_bins)}')


def print_areas_using_sampling():
    """Uses sampling method to print the areas of the functions"""
    # ------- Part A ---------
    num_samples = 10**4
    current_sum = 0
    for _ in range(num_samples):
        x = np.random.uniform(0, np.pi)
        current_sum += sqrt_sin(x)
    print(f'Area of √sin(x) over [0, pi] by sampling is: {current_sum / num_samples:.3f}')


    # ------- Part B ---------
    num_samples = 10**4
    current_sum = 0
    for _ in range(num_samples):
        x = abs(np.random.normal(0, 1/np.sqrt(2)))
        x = np.clip(x, 0, np.pi)
        current_sum += sqrt_sin_exp_minus_x2(x)
    print(f'Area of √sin(x)exp(-x2) over [0, pi] by sampling is: {current_sum / num_samples:.3f}')


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
    exp_minus_x2 = np.exp(-1 * x**2)

    plt.plot(x, sqrt_sin_x * exp_minus_x2)
    plt.title('x vs √sin(x)exp(-x2)')
    plt.xlabel('x')
    plt.ylabel('√sin(x)exp(-x2)')
    plt.savefig('figs/b.png')
    plt.clf()    


if __name__ == "__main__":
    pathlib.Path(f'{pathlib.Path.cwd()}/figs/').mkdir(parents=True, exist_ok=True)
    main()