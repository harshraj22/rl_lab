import sys
from PIL import Image, ImageDraw
import numpy as np
from numpy.typing import ArrayLike
from prettytable import PrettyTable
from typing import Dict, List

sys.path.insert(0, '../')
from base.iteration_env import IterationEnv

class PreserveEnvStateManager:
    """A context manager to preserve the gym environment state. The gym
    environment should have the same state after the context manager exits as
    the one it had when the context manager entered."""

    def __init__(self, env: IterationEnv) -> None:
        self.env = env

    def __enter__(self):
        self._env_state = self.env.state
        return self.env

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.env.state = self._env_state


def numpy_to_string(array: ArrayLike, title='Value Iteration: Values') -> str:
    """Convert a numpy array to a string."""
    
    table = PrettyTable()
    table.title = title
    table.add_rows(np.round(array, 2))
    return table.get_string()


def string_to_images(string: str, timestamp: int = 0, total: int = None):
    """Convert a string to an image."""
    # https://stackoverflow.com/a/17856617/10127204
    img = Image.new('RGB', (300, 180))
    d = ImageDraw.Draw(img)
    d.text((20, 2), string, fill=(255, 0, 0))
    d.text((20, 150), f'Iteration: {timestamp} / {total}', fill=(255, 0, 0))

    return img


def numpy_to_gif(array: ArrayLike, filename: str, title: str) -> None:
    """Convert a numpy array to a gif file."""
    array = [np.insert(values, 9, 0).reshape((4, 4)) for values in array]

    strings = [numpy_to_string(arr, title) for arr in array]
    frames = [string_to_images(string, index, len(strings)) for index, string in enumerate(strings)]
    # save the frames as GIF
    frames[0].save(filename, save_all=True, append_images=frames[1:], optimize=False, duration=300, loop=0)


def dict_to_string(policy: Dict[int, str]) -> str:
    """Convert a dict to a string."""
    table = PrettyTable()
    table.title = 'Policy Iteration: Policies'
    
    for i in range(4):
        table.add_row([policy[i*4+j] for j in range(4)])
    return table.get_string()


def dict_to_gif(policies: List[Dict[int, str]], filename: str) -> None:
    """Convert a dict to a gif file."""
    policies = [{**policy, **{9: 'X'}} for policy in policies]

    # strings = [numpy_to_string(arr) for arr in array]
    strings = [dict_to_string(policy) for policy in policies]
    frames = [string_to_images(string, index, len(strings)) for index, string in enumerate(strings)]
    # save the frames as GIF
    frames[0].save(filename, save_all=True, append_images=frames[1:], optimize=False, duration=300, loop=0)