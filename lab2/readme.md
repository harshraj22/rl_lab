## Study of Various Multi Arm Bandits algorithms, and affect of various parameters on final Regret


<img src="https://user-images.githubusercontent.com/46635452/152643153-ab55f9b3-c008-43d4-88e4-5b1d76244aab.png">


To run the code:         
1. install the dependencies using `pip install -r requirements.txt`
2. run the code using `python runner.py`

Note: The script uses [hydra](https://github.com/facebookresearch/hydra) for configuration management.
To alter the configuration settings, either edit the `conf/config.yaml` file or use the command line as `python3 runner.py seed=5`. The visualization is done using [wandb](https://wandb.ai/harshraj22/multi_arm_bandit)


See the [contributing](docs/contributing.md) file for more details regarding extending the code or adding more Agents.


Directory Structure:
```
.
├── base                       <- All base classes. Any extension in code should extend the subsequent base class
│   ├── arm_reward_initilizer.py
│   ├── multi_arm_bandit_agent.py
│   └── reward_distribution.py
├── conf                         <- Config files
│   └── config.yaml
├── data_loader                  <- classes for creating the MultiArmBandit environment      
│   ├── bandit_arm_reward_initializer.py
│   ├── environments.py
│   └── probablistic_reward_distributions.py
├── docs                         <- Technical docs for getting started/ contributing
│   └── contributing.md
├── models                       <- All the agents/ algos to solve MultiArmBandit
│   ├── __init__.py
│   ├── epsilon_greedy.py
│   ├── reinforce.py
│   ├── softmax.py
│   ├── thompson_sampling.py
│   └── ucb.py
├── outputs                     <- output file corresponding to each run (created by hydra)  
├── ques                        <- Problem statement and related docs
│   ├── auer.pdf
│   ├── banditsComparision.pdf
│   ├── bayesNormal.pdf
│   └── ques.pdf
├── readme.md
├── requirements.txt
├── runner.py                   <- Entry point of the script    
└── utils                       <- Various utility functions and classes
    └── utils.py
```

The api design has been created to make it easy to extend. In order to add a new agent, 
```python
# Create a new agent, it must inherit from MultiArmBanditAgent class

from base.multi_arm_bandit_agent import MultiArmBanditAgent
import numpy as np

class NewAgent(MultiArmBanditAgent):
# unlike PyTorch's models, you must define a few methods in the agent class, namely forward(), update_mean()
    def __init__(self, num_arms: int):
        super(NewAgent, self).__init__()
        self.num_arms = num_arms

    # forward() is the method that is called to make a prediction. It takes as input the state and returns 
    # the action
    def forward(self, state):
        # for now, let us select actions randomly
        return np.random.randint(0, self.num_arms)

    # update_mean() is the method that is called to update the underlying statistics of the reward distribution. 
    # It could be understood as the backward pass in case of a neural network which updates the underlying 
    # weights for better prediction
    def update_mean(self, reward):
        pass

```

The environment and interactions with agent have also been designed in a way that is familiar to reinforcement learning practitioners.

```python
from data_loader.environments import MultiArmBanditEnvironment
from data_loader.bandit_arm_reward_initializer import BanditArmRewardInitializer

# create a Multi Arm Bandit environment, with underlying reward distribution of each arm 
# following Bernoulli distribution
env = MultiArmBanditEnvironment(
    arm_initializer=BanditArmRewardInitializer('Bernoulli'),
    num_arms=3,
    total_timesteps=1000
    )
agent = NewAgent()

obs = env.reset()
for _ in range(episode_length):
    # agent selects an action
    action = agent(obs)
    # it interacts with the environment, and receives the reward
    obs, reward, done, info = env.step(action)
    
    # agent updates its knowledge about the environment, in order to select better actions in the future
    agent.update_mean(action, reward)

```
