# custom_env.py
import gymnasium as gym
import numpy as np
import os
import torch
import random

def make_vec_envs(n_envs, seed=None, **environment_kwargs):
    from stable_baselines3.common.vec_env import DummyVecEnv

    def set_random_seed(seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    if seed is not None:
        set_random_seed(seed)

    def make_env(rank, **env_kwargs):
        def _init():
            # Create a dictionary with default parameters
            gym_kwargs = {
                "continuous": env_kwargs.get("continuous", True),
                "gravity": env_kwargs.get("gravity", -10.0),
                "enable_wind": env_kwargs.get("enable_wind", False),
                "wind_power": env_kwargs.get("wind_power", 0.0),
                "turbulence_power": env_kwargs.get("turbulence_power", 0.0),
                "render_mode": env_kwargs.get("render_mode", "rgb_array")
            }
            
            env = gym.make("LunarLander-v2", **gym_kwargs)
            
            if seed is not None:
                env.reset(seed=seed + rank)
                env.action_space.seed(seed + rank)
                env.observation_space.seed(seed + rank)
            return env
        return _init

    return DummyVecEnv([make_env(i, **environment_kwargs) for i in range(n_envs)])