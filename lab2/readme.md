


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