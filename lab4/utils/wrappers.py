import gym

class LinearEnvWrapper(gym.Wrapper):
    """.
    A wrapper for the LinearEnv environment that adds the following:
    - Boost up the reward recieved on reaching the terminal state.
    - Remove the sparsity of rewards, by adding the eucledian distance between
        the current and goal state as intermediate rewards.
    """
    def __init__(self, env) -> None:
        super().__init__(env)
        self.env = env
        self._time = 1

    def step(self, action: int):
        cur_state = self.env.state
        state, reward, done, info = self.env.step(action)
        self._time += 1
        reward = reward * 100 + (state - cur_state)
        return state, reward, done, info