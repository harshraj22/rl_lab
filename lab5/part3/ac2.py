
import gym
import sys 

sys.path.append('/root/stb/')

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

import wandb
import os


wandb.init(
    project="Function Approximation", entity="harshraj22", mode='online'
)

# Parallel environments
env = make_vec_env("CartPole-v1", n_envs=1)

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=15000, wandb_obj = wandb)
model.save("a2c_cartpole")

del model # remove to demonstrate saving and loading

model = A2C.load("a2c_cartpole")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
