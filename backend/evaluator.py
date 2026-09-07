# evaluator.py

import os
import argparse
import yaml
import json
import gc
import random
import torch
import torch.nn as nn
import numpy as np
import stable_baselines3
import sb3_contrib
import logging

from configs.env_setup import train_environment, eval_environment, user_record_video
from configs.user_evaluation import user_evaluate_policy

def _run_paths(study_name: str):
    """
    Build per-user run paths: run/<uid>/<study_name>/...
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    uid = os.environ.get("PARTICIPANT_ID", "anon")
    root = os.path.join(script_dir, "run", uid, study_name)
    return {
        "root": root,
        "logs": os.path.join(root, "evaluators_logs"),
        "results": os.path.join(root, "results"),
        "models": os.path.join(root, "results", "models_saved"),
        "videos": os.path.join(root, "results", "videos_saved"),
    }

def setup_logger(study_name):
    """
    Logger writes to:
      run/<uid>/<study_name>/logs/evaluator_<TRIAL_ID>.log
    """
    trial_id = os.environ.get("TRIAL_ID", "unknown_trial")
    p = _run_paths(study_name)
    os.makedirs(p["logs"], exist_ok=True)
    log_file = os.path.join(p["logs"], f"evaluator_{trial_id}.log")

    logger = logging.getLogger("Evaluator")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicated handlers on re-import
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger


def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    """
    Evaluator script:
    1) Reads config.yaml to get general training settings.
    2) Reads hyperparameters from HYPERPARAMS_FILE (set by orchestrator).
    3) Trains a RL model for the full number of timesteps with no intermediate eval.
    4) Evaluates the policy once at the end.
    5) Writes the final objective value to RESULT_FILE as JSON.
    """
    try:
        parser = argparse.ArgumentParser(description="Run a single RL training job.")
        parser.add_argument("config_file_path", type=str, help="YAML config file path.")
        args = parser.parse_args()

        # 1) Load the main config.
        with open(args.config_file_path, 'r') as f:
            config = yaml.safe_load(f)

        study_name = config["configuration"]["study_name"]

        logger = setup_logger(study_name)

        logger.info(f"Evaluator started with config file: {args.config_file_path}")
        logger.info("Main configuration loaded successfully.")

        # Log key environment variables and show the bash command to recreate the run.
        env_vars_used = {
            "HYPERPARAMS_FILE": os.environ.get("HYPERPARAMS_FILE", ""),
            "TRIAL_ID": os.environ.get("TRIAL_ID", ""),
            "RESULT_FILE": os.environ.get("RESULT_FILE", ""),
            "PARTICIPANT_ID": os.environ.get("PARTICIPANT_ID", "anon"),
        }
        logger.info("Environment variables used:\n%s", json.dumps(env_vars_used, indent=4))
        bash_command = (
            f"HYPERPARAMS_FILE='{env_vars_used['HYPERPARAMS_FILE']}' "
            f"TRIAL_ID='{env_vars_used['TRIAL_ID']}' "
            f"RESULT_FILE='{env_vars_used['RESULT_FILE']}' "
            f"PARTICIPANT_ID='{env_vars_used['PARTICIPANT_ID']}' "
            f"python {os.path.basename(__file__)} {args.config_file_path}"
        )
        logger.info("To recreate this run, use the following bash command:\n%s", bash_command)

        # 2) Read hyperparams from JSON passed by orchestrator.
        hyperparams_file = os.environ.get("HYPERPARAMS_FILE", None)
        if hyperparams_file is None or not os.path.exists(hyperparams_file):
            logger.error("No valid HYPERPARAMS_FILE found in environment variables.")
            raise ValueError("No valid HYPERPARAMS_FILE found in environment variables.")

        with open(hyperparams_file, 'r') as hp_file:
            param_values = json.load(hp_file)
        logger.info(f"Hyperparameters loaded from file: {hyperparams_file}")

        # 3) Gather relevant training settings
        seed = config["agent_training_configuration"]["random_seed"]
        if seed == 'None' or seed == 'none':
                seed = None

        if seed is not None:
            set_random_seed(seed)

        logger.info(f"Random seed set to: {seed}")

        max_episode_steps = config["agent_training_configuration"]["max_number_steps_per_episode"]
        batch_size = config["agent_training_configuration"]["training_batch_size"]
        n_envs_train = config["agent_training_configuration"]["number_environments_for_training"]
        sb3_alg = config["agent_training_configuration"]["stable_baselines_algorithm"]
        sb3_policy_type = config["agent_training_configuration"]["stable_baselines_policy"]
        num_training_timesteps = config["agent_training_configuration"]["number_training_timesteps"]

        # Additional environment kwargs from param_values if any are "environment_kwargs"
        environment_kwargs = {}
        policy_kwargs = {}
        normal_params = {}

        for k, v in param_values.items():
            # Sanitize the value if it's a string
            if isinstance(v, str):
                v = v.lower().replace(" ", "_").rstrip(" _-:.")

            # Decide if it's environment-related or model param
            param_def = config["parameters_bounds"][k]
            ptype = param_def.get("type", "default")

            if ptype == "environment_kwargs":
                environment_kwargs[k] = v
            elif ptype == "policy_kwargs":
                policy_kwargs[k] = v
            else:
                normal_params[k] = v

        # Build final SB3 model kwargs
        model_kwargs = {
            "n_steps": max_episode_steps,
            "batch_size": batch_size,
            "policy_kwargs": {},
            "verbose": False,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "seed": seed
        }

        # Merge normal_params into model_kwargs
        for k, v in normal_params.items():
            model_kwargs[k] = v

        if "activation_fn" in policy_kwargs:
            act_fn_val = policy_kwargs.pop("activation_fn")
            model_kwargs["policy_kwargs"]["activation_fn"] = nn.ReLU if act_fn_val > 0.5 else nn.Tanh

        net_arch = []
        if "policy_arch_num_layers" in policy_kwargs and "policy_arch_num_neurons" in policy_kwargs:
            num_layers = policy_kwargs.pop("policy_arch_num_layers")
            num_neurons = policy_kwargs.pop("policy_arch_num_neurons")
            net_arch = [num_neurons] * num_layers
        if len(net_arch) > 0:
            model_kwargs["policy_kwargs"]["net_arch"] = net_arch

        logger.debug(f"Model kwargs constructed: {model_kwargs}")

        # 4) Create training and eval environments
        train_env = train_environment(seed, n_envs_train, **environment_kwargs)
        eval_env_ = eval_environment(seed, **environment_kwargs)
        logger.info("Training and evaluation environments created.")

        # 5) Create and train the model
        algorithm_class = getattr(stable_baselines3, sb3_alg)
        model = algorithm_class(sb3_policy_type, train_env, **model_kwargs)
        logger.info(f"Model created using algorithm: {sb3_alg} with policy type: {sb3_policy_type}")

        # Train for the full number of timesteps in one go (no intermediate eval):
        model.learn(total_timesteps=num_training_timesteps, progress_bar=False)
        logger.info(f"Model training completed for {num_training_timesteps} timesteps.")

        # 6) Evaluate once at the end
        with torch.no_grad():
            obj_value, extra_info = user_evaluate_policy(eval_env_, model)
        logger.info(f"Policy evaluation completed with objective value: {obj_value}")

        # 7) Save the model and evaluation extra info to the NEW per-user tree
        trial_id = os.environ.get("TRIAL_ID", "unknown_trial")
        p = _run_paths(study_name)
        os.makedirs(p["models"], exist_ok=True)
        os.makedirs(p["videos"], exist_ok=True)

        # NOTE: trial_id already looks like "trial_3", so don't prepend "trial_" again.
        model_filename = os.path.join(p["models"], f"{trial_id}_model.zip")
        model.save(model_filename)
        logger.info(f"Model saved to: {model_filename}")

        if extra_info is not None:
            for k, v in extra_info.items():
                if isinstance(v, np.float32):
                    extra_info[k] = float(v)

            logger.info(f"Extra info: {extra_info}")
            extra_info_file = os.path.join(p["models"], f"{trial_id}_eval_extra_info.json")
            with open(extra_info_file, "w") as fi:
                json.dump(extra_info, fi, indent=4)
            logger.info(f"Evaluation extra info saved to: {extra_info_file}")

        train_env.close()
        eval_env_.close()
        logger.info("Training and evaluation environments closed.")

        del model
        del train_env
        del eval_env_
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Cleaned up model and memory.")

        # Record video to NEW per-user path
        logger.info("Recording evaluation video(s)...")
        user_record_video(
            seed,
            model_filename,
            p["videos"],       # <<=== NEW location: run/<uid>/<study>/results/videos_saved
            trial_id,
            environment_kwargs,
            num_episodes=20,
            video_length=10000
        )

        # 8) Write the final objective to RESULT_FILE (this is already set by the orchestrator)
        result_file = os.environ.get("RESULT_FILE", None)
        if result_file is not None:
            with open(result_file, "w") as rf:
                json.dump({"objective_value": float(obj_value)}, rf)
            logger.info(f"Final objective value written to: {result_file}")
        else:
            logger.warning("RESULT_FILE environment variable not set; printing objective value instead.")
            print(json.dumps({"objective_value": float(obj_value)}))

    except Exception as e:
        logger.exception("An error occurred during evaluation:")
        raise

if __name__ == "__main__":
    main()
