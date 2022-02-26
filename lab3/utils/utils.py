import gym

class PreserveEnvStateManager:
    """A context manager to preserve the gym environment state. The gym
    environment should have the same state after the context manager exits as
    the one it had when the context manager entered."""

    def __init__(self, env: gym.Env) -> None:
        pass