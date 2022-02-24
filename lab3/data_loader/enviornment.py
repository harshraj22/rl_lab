import gym
import numpy as np

class grid_world_env(gym.Env):
    metadata = {'render.modes': ['human']}
    num_env = 0

    def __init__(self,grid_file:str, start_state: list = None):
        self.action = ['up','down','right','left']
        self.action_space = gym.spaces.Discrete(4)
        self.action_pos_dict = {'up':[-1,0],'down':[1,0],'right':[0,1],'left':[0,-1]}

        #observation space
        self.obs_shape = [123,128,3]
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float32)

        #initializing grid
        self.grid_file = grid_file
        self.grid = self._read_grid_file()
        self.obstacles = []

        #agent initial position
        if start_state is None:
            self.agent_state = self.initialize_agent_start_position()
        else:
            self.agent_state = start_state

    def initialize_agent_start_position(self):
        x, y = np.random.randint(self.grid.shape[0]), np.random.randint(self.grid.shape[1])
        return x,y
    
    def make_obstacle(self,obstacles: list = []) -> None:
        self.obstacles = obstacles

    def _read_grid_file(self):
        with open(self.grid_file,'r') as f:
            grid_lines = f.readlines()
        grid = np.array(list(map(lambda x: list(map(lambda y: int(y), x.split(' '))), grid_lines)))
        return grid
        
    def reset(self, start_state:list = None):
        if start_state is None:
            start_state = self.initialize_agent_start_position()
        self.agent_state = start_state

    def is_legal_action(self,action):
        future_state = [self.agent_state[0] + self.action_pos_dict[action][0], self.agent_state[1] + self.action_pos_dict[action][1]]
        # print(self.obstacles, future_state, future_state in self.obstacles)
        if future_state[0]>=0 and future_state[0]<self.grid.shape[0] and future_state[1]>=0 and future_state[1]<self.grid.shape[1] and future_state not in self.obstacles:
            return True
        return False


    def step(self, action: str):
        if self.is_legal_action(action):
            self.agent_state = self.agent_state[0] + self.action_pos_dict[action][0] , self.agent_state[1] + self.action_pos_dict[action][1]
            print(self.agent_state[0],self.agent_state[1])
            return self.grid[self.agent_state[0],self.agent_state[1]]
        return None

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
            
if __name__ == '__main__':
    env = grid_world_env('map.txt',[0,0])
    print(env.grid, env.agent_state)
    env.make_obstacle([[0,1]])
    # print(env.obstacles)
    print(env.step('down'))
    print(env.step('right'))
    print(env.step('right'))
    print(env.step('right'))
    print(env.reset())