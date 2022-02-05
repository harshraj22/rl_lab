python3 runner.py  wandb_tracking=online agent.type=ucb
python3 runner.py  wandb_tracking=online agent.type=eps_greedy
python3 runner.py  wandb_tracking=online agent.type=softmax
python3 runner.py  wandb_tracking=online agent.type=reinforce
python3 runner.py  wandb_tracking=online agent.type=reinforce agent.reinforce.baseline=False
python3 runner.py  wandb_tracking=online agent.type=thompson_sampling