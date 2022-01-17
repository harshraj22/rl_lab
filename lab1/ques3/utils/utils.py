import numpy as np
from typing import Callable
import logging
import sys
import unittest


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

np.set_printoptions(linewidth=np.inf)


def sqrt_sin_exp_minus_x2(x: float) -> float:
    return np.sqrt(np.sin(x)) * np.exp(-x**2)


def sqrt_sin(x: float) -> float:
    return np.sqrt(np.sin(x))


def calculate_area(value_func: Callable, num_bins: int = 20_000) -> float:
    """Calculates the approximate area swapped by value_func over the domain [0, pi]. Divides
    the domain into num_bins bins and calculates the area of each rectangle.

    Args:
        value_func (Callable): Function whose area is to be calculated.
        num_bins (int, optional): Number of bins in which the domain is to be
          divided. Defaults to 20_000.

    Returns:
        float: The area under the curve created by value_func in domain [0, pi].
    """

    bins = np.linspace(start=0, stop=np.pi, num=num_bins+1)
    values = np.zeros(num_bins)

    for index, value in enumerate(bins[1:], start=0):
        values[index] = value_func(value) * np.pi / num_bins

    return values.sum()


class TestCalculateArea(unittest.TestCase):
    def test_constant_function(self):
        const_func = lambda x: 2
        self.assertAlmostEqual(calculate_area(const_func, num_bins=20_000), 2 * np.pi)


class TestSqrtSin(unittest.TestCase):
    def test_multiples_of_pi(self):
        self.assertAlmostEqual(sqrt_sin(0), 0)
        self.assertAlmostEqual(sqrt_sin(np.pi), 0)
        self.assertAlmostEqual(sqrt_sin(np.pi / 2), 1)


class TestSqrtSinExpMinusX2(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(sqrt_sin_exp_minus_x2(0), 0)
        self.assertAlmostEqual(sqrt_sin_exp_minus_x2(np.pi), 0)


if __name__ == "__main__":
    unittest.main()