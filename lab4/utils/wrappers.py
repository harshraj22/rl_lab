import gym

class FrozenLakeWrapper(gym.Wrapper):
    """
    CAUTION:
        This is wrong wrapper.
    A wrapper for the FrozenLake environment that adds the following:
    Removes the issue of sparse reward setting, by giving a small reward for each
    timestep that the agent is alive.
    """
    def __init__(self, env) -> None:
        super().__init__(env)
        self.env = env
        self._time = 1

    def step(self, action: int):
        state, reward, done, info = self.env.step(action)
        self._time += 1
        reward = reward * 10 + self._time / 200
        return state, reward, done, info