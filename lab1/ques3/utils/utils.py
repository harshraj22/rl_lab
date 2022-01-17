import numpy as np
from typing import Callable


def sqrt_sin(x: float) -> float:
    return np.sqrt(np.sin(x))


def calculate_area(value_func: Callable, num_bins: int = 200, num_samples: int = 10_000):
    """Calculates the approximate area swapped by value_func over the domain [0, pi]. Divides
    the domain into num_bins bins and samples num_samples points in each bin. The

    Args:
        value_func (Callable): Function whose area is to be calculated.
        num_bins (int, optional): Number of bins in which the domain is to be 
          divided. Defaults to 200.
        num_samples (int, optional): Number of samples to draw from uniform 
          distribution in order to make it a good approximation. Defaults to 10_000.

    Returns:
        float: The area under the curve created by value_func in domain [0, pi].
    """

    bins = np.linspace(start=0, stop=np.pi, num=num_bins+1)
    values = np.zeros(num_samples)

    for index in range(num_samples):
        random_num = np.random.uniform(low=0, high=np.pi)
        bin_index = np.digitize(random_num, bins, right=True)
        values[index] = value_func(bin_index * np.pi / num_bins) * np.pi / num_bins

    return values.mean()
