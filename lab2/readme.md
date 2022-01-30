## Study of Various Multi Arm Bandits algorithms, and affect of various parameters on final Regret


<!- Plots to be added here ->


To run the code:         
1. install the dependencies using `pip install -r requirements.txt`
2. run the code using `python runner.py`

Note: The script uses [hydra](https://github.com/facebookresearch/hydra) for configuration management.
To alter the configuration settings, either edit the `conf/config.yaml` file or use the command line as `python3 runner.py seed=5`.


See the [contributing](docs/contributing.md) file for more details regarding extending the code or adding more Agents.


Directory Structure:
```
.
├── base                         <- All base classes are present. Any extension in code should extend the subsequent base class
│   ├── arm_reward_initilizer.py
│   ├── multi_arm_bandit_agent.py
│   └── reward_distribution.py
├── conf                         <- Config files
│   └── config.yaml
├── data_loader                  <- classes for creating the MultiArmBandit environment      
│   ├── __pycache__
│   │   └── environments.cpython-39.pyc
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