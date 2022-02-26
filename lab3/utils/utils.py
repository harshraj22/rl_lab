import sys

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