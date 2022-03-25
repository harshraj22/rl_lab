from gym.envs.registration import register

register(id="SimpleLinear-v0", entry_point="data_loader.environments:LinearEnv")
