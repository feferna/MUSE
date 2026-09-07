import os
import glob
import numpy as np
import random
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor, VecVideoRecorder

from configs.environment import make_vec_envs


def train_environment(seed, n_train_envs, **environment_kwargs):
    torch.set_num_threads(1)
    
    # Configure training environment
    train_env = make_vec_envs(n_train_envs, seed=seed, **environment_kwargs)
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True)

    return train_env

def eval_environment(seed, **environment_kwargs):
    torch.set_num_threads(1)

    if seed is not None:
        eval_env = make_vec_envs(1, seed=(seed+100), **environment_kwargs)
    else:
        eval_env = make_vec_envs(1, seed=None, **environment_kwargs)

    return eval_env

def user_record_video(seed, model_path, video_folder, trial_id, environment_kwargs, num_episodes=20, video_length=10000):
    torch.set_num_threads(1)

    if seed is not None:
        eval_env = make_vec_envs(1, seed=(seed+10108), **environment_kwargs)
    else:
        eval_env = make_vec_envs(1, seed=None, **environment_kwargs)
    
    eval_env = VecMonitor(eval_env)

    model = PPO.load(model_path)

    if seed is not None:
        model.set_random_seed(seed=seed)

    video_recorder = VecVideoRecorder(
        eval_env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=trial_id
    )

    current_episode = 0

    obs = video_recorder.reset()
    while current_episode < num_episodes:
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = video_recorder.step(action)
            
        current_episode += 1

    video_recorder.close()
    eval_env.close()

    # Rename the video file to 'trial_X_video.mp4'
    video_pattern = os.path.join(video_folder, f"{trial_id}*.mp4")
    video_files = glob.glob(video_pattern)

    if video_files:
        original_video_path = video_files[0]
        new_video_path = os.path.join(video_folder, f"{trial_id}_video.mp4")
        os.rename(original_video_path, new_video_path)
        print(f"Video saved as: {new_video_path}")
    else:
        print(f"No video file found with prefix '{trial_id}' in folder '{video_folder}'")
