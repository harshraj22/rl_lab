## Project overview:
We define base classes to be inherited from for various components of the experiement so that it becomes easy to plug and extend the code for new components.
- #### Agents
    - `models` dir contains all the implementations of the agents.
    - All the agents must inherit from the base class `MultiArmBanditAgent` defined in `base/multi_arm_bandit_agent.py`. This is to maintain consistency in the code.
- #### Environment
    - `environments` dir contains all the implementations of the environments. For now, we have only one environment, `MultiArmBanditEnvironment`.
    - All the environments must inherit from the base class `MultiArmBanditEnvironment` defined in `base/multi_arm_bandit_environment.py`. This is to maintain consistency in the code.
- #### Initializing the underlying reward distribution
    - `data_loader/probablistic_reward_distributions.py` contains all the implementations of the reward distributions.
    - All the reward distributions must inherit from the base class `RewardDistribution` defined in `base/reward_distribution.py`. This is to maintain consistency in the code.
    - Once a new reward distribution is added, it must be added to the `data_loader/bandit_arm_reward_initializer.py` file.
    - The environment accepts a callable that returns the underlying reward distribution for its arms.

     

## Contributing/ extending the code:

### Adding a new reward distribution
In order to add a new reward distribution, you need to:
1. Create a new class that inherits from the `RewardDistribution` class. It should define how the reward is distributed for each arm.
2. Update the `__call__` method of the `BanditArmRewardInitializer` class to include the new reward distribution.
3. Use the new reward distribution as follows:
```python
from data_loader.environments import MultiArmBanditEnvironment
from data_loader.bandit_arm_reward_initializer import BanditArmRewardInitializer

env = MultiArmBanditEnvironment(arm_initializer=BanditArmRewardInitializer('new-distribution'))
```