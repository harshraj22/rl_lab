import gym
import numpy as np
from gym import spaces


class SnakeAndLadderEnv(gym.Env):
    def __init__(self, current_state=0, num_states=9, dead_states=(2, 4), view=None) -> None:
        super().__init__()
        self.current_state = current_state
        self.num_states = num_states
        self.dead_states = dead_states
        self.view = view if view else (self.num_states, )

    def reset(self):
        self.current_state = 0
        return self.current_state

    def step(self, action):
        self.current_state = np.clip([self.current_state + action], 0, self.num_states - 1).item()
        return self.current_state, int(self.current_state == self.num_states - 1), bool(self.current_state in [self.num_states-1, *self.dead_states]), {}
        
    def render(self):
        states = np.chararray(shape=(self.num_states,), unicode=True)
        for i in range(self.num_states):
            states[i] = str(i)
        states[self.current_state] = '#'
        states[list(self.dead_states)] = '_'
        print(states.reshape(self.view))
        print(f'#: current state \n_: dead state\n')

    def close(self):
        pass
