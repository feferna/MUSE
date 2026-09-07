# EvaluationToggleDesign.py - Backend Sample
# This is the same file as in frontend/public/files/
# Copy of frontend file for backend reference

import numpy as np
import json


def user_evaluate_policy(model, env, n_eval_episodes=10, deterministic=True):
    """
    Evaluate a trained model on the toggle design task.

    Args:
        model: The trained RL model
        env: The toggle design environment
        n_eval_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic actions

    Returns:
        dict: Evaluation metrics including user satisfaction, usability, and aesthetic scores
    """

    episode_rewards = []
    episode_lengths = []
    satisfaction_scores = []

    for episode in range(n_eval_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_satisfactions = []

        done = False
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=deterministic)

            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            # Track satisfaction throughout episode
            if hasattr(env, 'user_satisfaction'):
                episode_satisfactions.append(env.user_satisfaction)

            if truncated:
                done = True

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if episode_satisfactions:
            satisfaction_scores.append(np.mean(episode_satisfactions))

    # Calculate overall metrics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_satisfaction = np.mean(
        satisfaction_scores) if satisfaction_scores else 0.5

    # Calculate usability metrics based on toggle design parameters
    usability_score = calculate_usability_score(env)
    aesthetic_score = calculate_aesthetic_score(env)
    accessibility_score = calculate_accessibility_score(env)

    # Overall performance combines multiple factors
    overall_score = (
        mean_reward * 0.3 +
        mean_satisfaction * 0.25 +
        usability_score * 0.25 +
        aesthetic_score * 0.15 +
        accessibility_score * 0.05
    )

    evaluation_results = {
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'mean_satisfaction': float(mean_satisfaction),
        'usability_score': float(usability_score),
        'aesthetic_score': float(aesthetic_score),
        'accessibility_score': float(accessibility_score),
        'overall_score': float(overall_score),
        'n_episodes': n_eval_episodes,
        'mean_episode_length': float(np.mean(episode_lengths))
    }

    return evaluation_results


def calculate_usability_score(env):
    """Calculate usability score based on toggle design parameters."""
    score = 0.5  # Base score

    # Duration usability (optimal range: 200-400ms)
    duration = getattr(env, 'duration', 300)
    if 200 <= duration <= 400:
        score += 0.2
    elif 150 <= duration <= 600:
        score += 0.1
    else:
        score -= 0.1

    # Thumb ratio usability (optimal range: 0.7-0.8)
    thumb_ratio = getattr(env, 'thumb_ratio', 0.75)
    if 0.7 <= thumb_ratio <= 0.8:
        score += 0.15
    elif 0.65 <= thumb_ratio <= 0.85:
        score += 0.05

    # Border radius usability (moderate values preferred)
    border_radius = getattr(env, 'border_radius', 25)
    if 10 <= border_radius <= 30:
        score += 0.1
    elif border_radius > 40:
        score -= 0.05  # Too rounded can be confusing

    # Easing usability
    easing = getattr(env, 'easing', 'ease')
    easing_scores = {
        'ease': 0.1,
        'ease-out': 0.08,
        'ease-in-out': 0.06,
        'ease-in': 0.04,
        'linear': 0.02
    }
    score += easing_scores.get(easing, 0.0)

    return np.clip(score, 0.0, 1.0)


def calculate_aesthetic_score(env):
    """Calculate aesthetic appeal score based on design parameters."""
    score = 0.5  # Base score

    # Visual style aesthetic appeal
    visual_style = getattr(env, 'visual_style', 'flat-solid')
    style_scores = {
        'shadow-gradient': 0.2,
        'glow-glass': 0.15,
        'flat-solid': 0.12,
        'elevated-textured': 0.1,
        'outline-outlined': 0.08
    }
    score += style_scores.get(visual_style, 0.05)

    # Color scheme appeal
    color_scheme = getattr(env, 'color_scheme', 'modern')
    color_scores = {
        'modern': 0.15,
        'nature': 0.12,
        'sunset': 0.12,
        'ocean': 0.1,
        'enterprise': 0.08,
        'dark': 0.06,
        'neon': 0.04
    }
    score += color_scores.get(color_scheme, 0.05)

    # Border radius aesthetic (smooth curves preferred)
    border_radius = getattr(env, 'border_radius', 25)
    if 15 <= border_radius <= 35:
        score += 0.1
    elif 5 <= border_radius <= 45:
        score += 0.05

    return np.clip(score, 0.0, 1.0)


def calculate_accessibility_score(env):
    """Calculate accessibility score based on design parameters."""
    score = 0.5  # Base score

    # Duration accessibility (not too fast, not too slow)
    duration = getattr(env, 'duration', 300)
    if 250 <= duration <= 500:
        score += 0.2  # Good for users with motor difficulties
    elif duration < 150:
        score -= 0.1  # Too fast

    # Thumb ratio accessibility (larger targets better)
    thumb_ratio = getattr(env, 'thumb_ratio', 0.75)
    if thumb_ratio >= 0.75:
        score += 0.15  # Larger targets easier to hit
    elif thumb_ratio < 0.65:
        score -= 0.1  # Too small

    # Color scheme accessibility
    color_scheme = getattr(env, 'color_scheme', 'modern')
    accessible_schemes = ['modern', 'enterprise', 'dark']
    if color_scheme in accessible_schemes:
        score += 0.1
    elif color_scheme == 'neon':
        score -= 0.05  # Can be harsh on eyes

    # Visual style accessibility
    visual_style = getattr(env, 'visual_style', 'flat-solid')
    if visual_style in ['flat-solid', 'outline-outlined']:
        score += 0.05  # Clear, simple styles

    return np.clip(score, 0.0, 1.0)


def user_record_video(model, env, video_path, n_episodes=1):
    """Record video of the toggle design interaction."""

    interaction_log = {
        'video_path': video_path,
        'episodes': [],
        'toggle_design': {
            'duration': getattr(env, 'duration', 300),
            'border_radius': getattr(env, 'border_radius', 25),
            'thumb_ratio': getattr(env, 'thumb_ratio', 0.75),
            'visual_style': getattr(env, 'visual_style', 'flat-solid'),
            'color_scheme': getattr(env, 'color_scheme', 'modern'),
            'easing': getattr(env, 'easing', 'ease')
        }
    }

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_log = {
            'episode': episode,
            'interactions': [],
            'satisfaction_over_time': []
        }

        done = False
        step = 0
        while not done and step < 100:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            # Log interaction
            interaction = {
                'step': step,
                'action': int(action),
                'reward': float(reward),
                'toggle_state': bool(obs[0]) if len(obs) > 0 else False,
                'satisfaction': float(obs[1]) if len(obs) > 1 else 0.5
            }
            episode_log['interactions'].append(interaction)

            if hasattr(env, 'user_satisfaction'):
                episode_log['satisfaction_over_time'].append(
                    env.user_satisfaction)

            step += 1
            if truncated:
                done = True

        interaction_log['episodes'].append(episode_log)

    # Save interaction log as JSON
    log_path = video_path.replace('.mp4', '_interaction_log.json')
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(interaction_log, f, indent=2)
        print(f"Interaction log saved to: {log_path}")
    except Exception as e:
        print(f"Failed to save interaction log: {e}")

    return log_path
