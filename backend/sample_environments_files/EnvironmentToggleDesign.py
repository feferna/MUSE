# EnvironmentToggleDesign.py - Backend Sample
# This is the same file as in frontend/public/files/
# Copy of frontend file for backend reference

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ToggleDesignEnvironment(gym.Env):
    """
    Environment for optimizing toggle design parameters.
    This environment simulates user interactions with different toggle designs.
    """

    def __init__(self):
        super(ToggleDesignEnvironment, self).__init__()

        # Define action and observation space
        # Actions: 0 = interact with toggle, 1 = observe, 2 = rate design
        self.action_space = spaces.Discrete(3)

        # Observations: toggle state, user satisfaction, interaction metrics
        # Fields: toggle_state, user_satisfaction, duration, border_radius, thumb_ratio,
        # thumb_shape_encoded, visual_style_encoded, color_scheme_encoded, easing_encoded, step_progress
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(10,), dtype=np.float32)

        # Toggle design parameters (will be set from optimization)
        self.duration = 300  # Animation duration in ms
        self.border_radius = 25  # Border radius in px
        self.thumb_ratio = 0.75  # Thumb size ratio
        self.thumb_shape = "circle"  # Thumb shape
        self.visual_style = "shadow-gradient"  # Visual style
        self.color_scheme = "modern"  # Color scheme
        self.easing = "ease-out"  # Animation easing

        # Environment state
        self.current_step = 0
        self.max_steps = 100
        self.toggle_state = False
        self.user_satisfaction = 0.5

    def set_design_parameters(self, params):
        """Set the toggle design parameters from optimization"""
        self.duration = params.get('duration', 300)
        self.border_radius = params.get('border_radius', 25)
        self.thumb_ratio = params.get('thumb_ratio', 0.75)
        self.thumb_shape = params.get('thumb_shape', 'circle')
        self.visual_style = params.get('visual_style', 'shadow-gradient')
        self.color_scheme = params.get('color_scheme', 'modern')
        self.easing = params.get('easing', 'ease-out')

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        self.current_step = 0
        self.toggle_state = False
        self.user_satisfaction = 0.5

        return self._get_observation(), {}

    def step(self, action):
        """Execute one step in the environment"""
        self.current_step += 1

        # Simulate user interaction based on toggle design
        reward = self._calculate_reward(action)

        # Update toggle state based on action
        if action == 0:  # Interact with toggle
            self.toggle_state = not self.toggle_state
            # Calculate satisfaction based on design parameters
            self.user_satisfaction = self._calculate_satisfaction()

        elif action == 1:  # Observe
            # Observing doesn't change toggle state but provides feedback
            pass

        elif action == 2:  # Rate design
            # User provides explicit rating
            reward += self.user_satisfaction

        done = self.current_step >= self.max_steps
        truncated = False

        return self._get_observation(), reward, done, truncated, {}

    def _get_observation(self):
        """Get current observation"""
        obs = np.array([
            float(self.toggle_state),
            self.user_satisfaction,
            self.duration / 1000.0,  # Normalized duration
            self.border_radius / 50.0,  # Normalized border radius
            self.thumb_ratio,
            self._encode_thumb_shape(),
            self._encode_visual_style(),
            self._encode_color_scheme(),
            self._encode_easing(),
            self.current_step / self.max_steps,
        ], dtype=np.float32)
        return obs

    def _calculate_satisfaction(self):
        """Calculate user satisfaction based on design parameters"""
        # This is a simplified model - in real use, you'd have actual user data
        satisfaction = 0.5

        # Duration preference (moderate durations preferred)
        if 200 <= self.duration <= 400:
            satisfaction += 0.1
        elif self.duration < 150 or self.duration > 600:
            satisfaction -= 0.1

        # Border radius preference
        if 10 <= self.border_radius <= 30:
            satisfaction += 0.1

        # Thumb ratio preference
        if 0.7 <= self.thumb_ratio <= 0.8:
            satisfaction += 0.1

        # Visual style preferences (example preferences)
        style_scores = {
            "flat-solid": 0.8,
            "shadow-gradient": 0.9,
            "glow-glass": 0.6,
            "outline-outlined": 0.5,
            "elevated-textured": 0.7,
        }
        satisfaction += (style_scores.get(self.visual_style, 0.5) - 0.5) * 0.2

        # Thumb shape preferences (simpler shapes generally better for Duolingo)
        shape_scores = {
            "circle": 0.9,
            "square": 0.7,
            "diamond": 0.5,
            "hexagon": 0.4,
            "star": 0.3,
            "triangle": 0.4,
            "teardrop": 0.6,
            "bean": 0.5,
        }
        satisfaction += (shape_scores.get(self.thumb_shape, 0.5) - 0.5) * 0.1

        return np.clip(satisfaction, 0.0, 1.0)

    def _calculate_reward(self, action):
        """Calculate reward for the current action"""
        base_reward = 0.1

        # Reward based on user satisfaction
        satisfaction_reward = self.user_satisfaction * 0.5

        # Penalty for very long or short durations (usability)
        duration_penalty = 0
        if self.duration < 100 or self.duration > 800:
            duration_penalty = -0.2

        return base_reward + satisfaction_reward + duration_penalty

    def _encode_visual_style(self):
        """Encode visual style as a numerical value"""
        styles = ["flat-solid", "shadow-gradient", "glow-glass",
                  "outline-outlined", "elevated-textured"]
        try:
            return styles.index(self.visual_style) / len(styles)
        except ValueError:
            return 0.0

    def _encode_thumb_shape(self):
        """Encode thumb shape as a numerical value"""
        shapes = ["circle", "square", "diamond", "hexagon",
                  "star", "triangle", "teardrop", "bean"]
        try:
            return shapes.index(self.thumb_shape) / len(shapes)
        except ValueError:
            return 0.0

    def _encode_color_scheme(self):
        """Encode color scheme as a numerical value"""
        schemes = ["modern", "nature", "sunset",
                   "ocean", "dark", "neon", "enterprise"]
        try:
            return schemes.index(self.color_scheme) / len(schemes)
        except ValueError:
            return 0.0

    def _encode_easing(self):
        """Encode easing function as a numerical value"""
        easings = ["linear", "ease", "ease-in", "ease-out", "ease-in-out"]
        try:
            return easings.index(self.easing) / len(easings)
        except ValueError:
            return 0.0


def make_toggle_env():
    """Factory function to create toggle environment"""
    return ToggleDesignEnvironment()
