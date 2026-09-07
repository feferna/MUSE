import numpy as np
import torch

def user_evaluate_policy(eval_env, model):
    num_eval_episodes = 300

    # Use an alternative reward to evaluate the performance of the agent
    landings_count = 0
    rewards = []

    obs = eval_env.reset()
    for _ in range(num_eval_episodes):
        eval_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done, info = eval_env.step(action)
            
            eval_reward += rew

            if 'landed' in info[0] and info[0]['landed']:
                landings_count += 1
        
        rewards.append(eval_reward)

    extra_info = {"landing_success_rate": landings_count / num_eval_episodes}

    eval_env.close()
    del eval_env
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    extra_info['std_reward'] = std_reward

    return mean_reward, extra_info