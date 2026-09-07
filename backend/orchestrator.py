# orchestrator.py - The brain behind our design optimization system

import json  # make sure these are imported once at top
import subprocess
import sys
import os
import json
import yaml
import subprocess
import asyncio
import optuna
import logging
import shutil
from subprocess import Popen, PIPE
import math
import numpy as np
import pandas as pd

from gp_fitting import fit_all_gp

from evaluator_toggles import evaluate_toggle_design


class OptimizationOrchestrator:
    def __init__(self):
        """
        Sets up our optimization system - think of this as the conductor of an orchestra
        where each instrument is a different trial running on potentially different GPUs.

        We'll track trials, handle user preferences, and manage the optimization process.
        """
        self.sources_to_use = None      # set after reading config
        self.wave_metric_values = {}    # always available; may remain empty

        # Clean up any leftover config files from previous runs
        files_to_delete = [
            "./configs/config.yaml",
            "./configs/environment.py",
            "./configs/user_evaluation.py"
        ]

        for file_path in files_to_delete:
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass  # No problem if the file isn't there

        self.task_name = None
        self.start_optimization_called = 0

        self.chart_mode = None
        self.chart_task = None
        self.toggle_mode = None

        self.g_min = -1.5
        self.g_max = 1.5

        # Keep track of what's running and where
        self.active_trials = []
        self.allocations = {}  # Maps GPUs to the processes running on them
        self.study = None

        # Storage for our trials and what the user thinks of them
        self.trials_dataset = pd.DataFrame(columns=["trial_number",
                                                    "parameters",
                                                    "simulated_based_value"])

        self.user_preferences_dataset = pd.DataFrame(columns=["chosen_trial_number",
                                                              "all_trial_numbers_shown"])

        self.trials_full_dataset = pd.DataFrame(columns=[
            "trial_number",
            "parameters",
            "simulated_based_objective",
            "human_objective",
            "combined_objective",
        ])

        # Keep track of which process belongs to which trial
        self.process_to_trialinfo = {}

        # Info about the current batch we're showing to the user
        self.current_batch_info = {
            "batch_in_progress": False,
            "trials": [],
            "message": "No batch has started yet."
        }

        # What the user selected as their preferred designs in this batch
        self.chosen_trials = None
        self.not_chosen_trials = None

        # Data about where our models disagree/aren't confident
        self.disagreement_data = {}
        self.uncertainty_data = {}

        self.stop_requested = False

        # user context (filled at start)
        self.user_id = "anon"

        # Initialize a basic logger that can be used before start_optimization
        self.logger = logging.getLogger("OptimizationOrchestrator")
        self.logger.setLevel(logging.INFO)

        # Add console handler if not already present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def _toggle_mode_folder(self):
        """
        Determine the toggle task 'mode' folder name.
        Prefer explicit sources (source1..3 -> self.sources_to_use) as 'a+b[+c]'.
        Fallback to provided toggle_mode/config or 'tool'.
        """
        # sources_to_use is set in start_optimization; initialized as None in __init__
        if getattr(self, "sources_to_use", None):
            # preserve order as defined by source1, source2, source3
            return "+".join(self.sources_to_use)
        # fallback to existing behavior
        return (
            (self.toggle_mode or None) or
            (self.config.get("configuration", {}).get("toggle_mode") if hasattr(self, "config") else None) or
            "tool"
        )

    def _paths(self):
        """Centralized per-user, per-study paths under run/."""
        base = os.path.abspath(os.getcwd())

        # Ensure we have required attributes
        user_id = getattr(self, "user_id", None) or "anon"
        study_name = getattr(self, "study_name", None) or "default_study"

        if getattr(self, "task_name", None) == "chart_reader":
            mode = self.chart_mode or self.config.get(
                "configuration", {}).get("chart_mode") or "tool"
            task = self.chart_task or self.config.get(
                "configuration", {}).get("chart_task") or "theme_parks"
            root = os.path.join(base, "run", user_id,
                                "chartist_task", mode, task)
        elif getattr(self, "task_name", None) == "toggle_task":
            mode = self._toggle_mode_folder()
            root = os.path.join(base, "run", user_id, "toggle_task", mode)
        else:
            root = os.path.join(base, "run", user_id, study_name)

        return {
            "root":     root,
            "logs":     os.path.join(root, "orchestrator_logs"),
            "results":  os.path.join(root, "results"),
            "db_dir":   os.path.join(root, "databases"),
            "datasets": os.path.join(root, "datasets"),
        }

    def _flush_datasets(self):
        """Write analysis tables every batch."""
        try:
            import pathlib
            p = self._paths()

            # Defensive check - ensure datasets path is valid
            datasets_path = p.get("datasets")
            if not datasets_path:
                self.logger.warning(
                    "Datasets path is None, skipping dataset flush")
                return

            pathlib.Path(datasets_path).mkdir(parents=True, exist_ok=True)

            if not self.trials_dataset.empty:
                self.trials_dataset.to_csv(os.path.join(
                    datasets_path, "trials.csv"), index=False)
                try:
                    self.trials_dataset.to_parquet(os.path.join(
                        datasets_path, "trials.parquet"), index=False)
                except Exception:
                    pass
            if not self.user_preferences_dataset.empty:
                self.user_preferences_dataset.to_csv(os.path.join(
                    datasets_path, "user_preferences.csv"), index=False)
                try:
                    self.user_preferences_dataset.to_parquet(os.path.join(
                        datasets_path, "user_preferences.parquet"), index=False)
                except Exception:
                    pass
            if not self.trials_full_dataset.empty:
                self.trials_full_dataset.to_csv(os.path.join(
                    datasets_path, "trials_full.csv"), index=False)
                try:
                    self.trials_full_dataset.to_parquet(os.path.join(
                        datasets_path, "trials_full.parquet"), index=False)
                except Exception:
                    pass
        except Exception as e:
            self.logger.warning(f"Error flushing datasets: {e}")
            # Don't raise the exception, just log it and continue

    def _log_trial_sources(self, tnum, r_dict, g_dict, f_val, disagreement_dict, uncertainty_dict):
        """
        Pretty, source-aware logging. If sources were explicitly provided in config,
        show each source's value; otherwise keep the legacy log line.
        """
        if not getattr(self, "sources_explicit", False):
            # Legacy log (unchanged)
            self.logger.info(
                f"Study told about trial {tnum} => f(x) = {f_val} "
                f"(sim_mu={r_dict.get(tnum,0.0)}, human_mu={g_dict.get(tnum,0.0)}, "
                f"disagreement={disagreement_dict.get(tnum,0.0):.2f}, "
                f"uncertainty={uncertainty_dict.get(tnum,0.0):.2f})"
            )
            return

        # Explicit sources: show exactly what was used
        parts = []
        if "llm" in self.sources_to_use:
            parts.append(f"llm={r_dict.get(tnum, 0.0):.4f}")
        if "human" in self.sources_to_use:
            parts.append(f"human={g_dict.get(tnum, 0.0):.4f}")
        if "wave" in self.sources_to_use:
            wv = self.wave_metric_values.get(tnum, None)
            parts.append(f"wave={wv:.4f}" if isinstance(
                wv, (int, float)) else "wave=—")

        # Say which fusion rule was used
        fusion = (
            "lambda-blend"
            if set(self.sources_to_use) == {"llm", "human"} and len(self.sources_to_use) == 2
            else "mean"
        )
        src_label = "+".join(self.sources_to_use)

        self.logger.info(
            f"Study told about trial {tnum} [sources:{src_label}; fusion:{fusion}] "
            f"=> f(x)={f_val:.4f} "
            f"({', '.join(parts)}; "
            f"disagreement={disagreement_dict.get(tnum,0.0):.2f}, "
            f"uncertainty={uncertainty_dict.get(tnum,0.0):.2f})"
        )

    def get_wave_metric_from_file(self, filepath):
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                v = data.get("wave_metric", None)
                return float(v) if v is not None else None
        except Exception:
            pass
        return None

    def record_trial_full(self, trial_number, parameters, simulated_value, human_value, combined_value, wave_metric=None):
        # store parameters as a JSON string to keep CSV clean
        try:
            params_json = json.dumps(parameters, sort_keys=True)
        except Exception:
            params_json = str(parameters)

        row = {
            "trial_number": trial_number,
            "parameters": params_json,
            "simulated_based_objective": float(simulated_value) if simulated_value is not None else None,
            "human_objective": float(human_value) if human_value is not None else None,
            "combined_objective": float(combined_value) if combined_value is not None else None,
        }

        if self._uses("wave"):
            row["wave_metric"] = float(
                wave_metric) if wave_metric is not None else None

        if self.trials_full_dataset.empty:
            self.trials_full_dataset = pd.DataFrame([row])
        else:
            self.trials_full_dataset = pd.concat(
                [self.trials_full_dataset, pd.DataFrame([row])],
                ignore_index=True
            )

    async def start_optimization(self):
        """
        The main event! This kicks off our optimization process.

        We'll create batches of designs, run them in parallel across GPUs, 
        ask the user for feedback, and use both simulation data and user preferences
        to guide the search toward better designs.
        """
        self.start_optimization_called += 1

        config_file_path = os.path.abspath("./configs/config.yaml")

        # First, let's load the user's configuration
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(
                f"The configuration file was not found: {config_file_path}")
        else:
            with open(config_file_path, 'r') as f:
                self.config = yaml.safe_load(f)

            self.stop_requested = False

            # Resolve user_id (config key 'participant_id' or env var)
            self.user_id = (
                self.config.get("configuration", {}).get("participant_id")
                or os.environ.get("PARTICIPANT_ID")
                or "anon"
            )
            # pass to child processes
            os.environ["PARTICIPANT_ID"] = self.user_id

            # Custom sources for the toggle case
            print(f"DEBUG: Full config = {self.config}")
            print(
                f"DEBUG: Configuration section = {self.config.get('configuration', {})}")

            def _norm_source(s):
                if not s:
                    return None
                s = str(s).strip().lower()
                if s in ("llm", "sim", "simulation", "model"):
                    return "llm"
                if s in ("human", "pref", "preference", "user"):
                    return "human"
                if s in ("wave", "color", "colour"):
                    return "wave"
                return None

            srcs_raw = []

            # First check for explicit source1/2/3 keys
            for key in ("source1", "source2", "source3"):
                v = (self.config.get(key) or self.config.get(
                    "configuration", {}).get(key))
                print(f"DEBUG: {key} = {v}")
                n = _norm_source(v)
                print(f"DEBUG: normalized {key} = {n}")
                if n and n not in srcs_raw:
                    srcs_raw.append(n)

            # If no explicit sources found, check toggle_mode for sources
            if not srcs_raw:
                toggle_mode = self.config.get(
                    "configuration", {}).get("toggle_mode", "")
                print(f"DEBUG: toggle_mode = {toggle_mode}")
                if toggle_mode:
                    # Normalize separators and split on non-alphanumeric characters
                    # Accept values like "llm+wave", "LLM+Human", "llm, wave", "llm wave"
                    import re
                    parts = [p for p in re.split(
                        r"[^A-Za-z0-9]+", str(toggle_mode)) if p]
                    parsed = []
                    for src in parts:
                        n = _norm_source(src)
                        if n and n not in parsed:
                            parsed.append(n)
                    if parsed:
                        srcs_raw = parsed
                        print(
                            f"DEBUG: Parsed sources from toggle_mode: {srcs_raw}")
                    else:
                        # Special-case: if toggle_mode is not the legacy 'tool', treat it as an explicit mode
                        if str(toggle_mode).strip().lower() not in ("tool", "default"):
                            print(
                                f"DEBUG: toggle_mode='{toggle_mode}' looks like an explicit mode but no known source tokens found")

            # True if user explicitly set any source key OR toggle_mode contains sources
            self.sources_explicit = len(srcs_raw) > 0
            if self.sources_explicit:
                self.logger.info(f"Explicit sources detected: {srcs_raw}")
            else:
                self.logger.info(
                    "No explicit sources found; using default sources")
            # If nothing provided, keep legacy default (LLM+Human)
            self.sources_to_use = srcs_raw if self.sources_explicit else [
                "llm", "human"]

            # Debug logging
            print(f"DEBUG: srcs_raw = {srcs_raw}")
            print(f"DEBUG: sources_explicit = {self.sources_explicit}")
            print(f"DEBUG: sources_to_use = {self.sources_to_use}")
            self.logger.info(
                f"Sources configuration: explicit={self.sources_explicit}, sources={self.sources_to_use}")

            def _uses(name: str) -> bool:
                result = name in self.sources_to_use
                print(f"DEBUG: _uses('{name}') = {result}")
                return result
            self._uses = _uses
            # End of sources for the toggle task

            self.chart_mode = self.config.get(
                "configuration", {}).get("chart_mode")
            self.chart_task = self.config.get(
                "configuration", {}).get("chart_task")
            self.toggle_mode = self.config.get(
                "configuration", {}).get("toggle_mode")

            # Pull out all the config values we need
            self.task_name = self.config["configuration"]["task_name"]
            self.batch_optim = self.config["configuration"]["n_designs"]
            self.num_gpus = self.config["parallellization"]["n_gpus"]
            self.num_parallel_evaluations = self.config["parallellization"]["n_processes"]
            self.study_name = self.config["configuration"]["study_name"]
            self.database_name = self.config["configuration"]["database_name"]
            self.continue_from_existing_database = self.config[
                "agent_training_configuration"]["continue_from_existing_database"]
            self.lambd = self.config["multi_surrogate_lambda"]
            # Precompute bounds and y-range for GP fitting
            self.param_bounds = {k: (v["start"], v["stop"])
                                 for k, v in self.config["parameters_bounds"].items()}
            self.y_min = self.config["y_min"]
            self.y_max = self.config["y_max"]

            # Set up our folder structure under run/<user_id>/<study>/
            p = self._paths()
            logs_folder = p["logs"]
            results_folder = p["results"]
            database_folder = p["db_dir"]

            if not self.continue_from_existing_database:
                # Start fresh - clear out previous results if we're not continuing
                shutil.rmtree(logs_folder, ignore_errors=True)
                shutil.rmtree(results_folder, ignore_errors=True)
                shutil.rmtree(database_folder, ignore_errors=True)
                shutil.rmtree(p["datasets"], ignore_errors=True)

                # And make fresh directories
                os.makedirs(logs_folder, exist_ok=True)
                os.makedirs(results_folder, exist_ok=True)
                os.makedirs(database_folder, exist_ok=True)

            os.makedirs(logs_folder, exist_ok=True)
            os.makedirs(results_folder, exist_ok=True)
            os.makedirs(database_folder, exist_ok=True)

            log_file_path = os.path.join(logs_folder, "orchestrator.log")
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            # Set up our logging system
            self.logger = logging.getLogger("OptimizationOrchestrator")
            # Catch everything down to DEBUG level
            self.logger.setLevel(logging.DEBUG)
            self.logger.handlers.clear()

            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Add console output too, but only once to avoid duplicate messages
            if self.start_optimization_called <= 1:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

            self.logger.debug(f"Root run folder: {p['root']}")
            self.logger.debug(f"Logs folder: {logs_folder}")

            # Get or create our Optuna study for tracking optimization
            self.study = self.get_or_create_study(self.study_name,
                                                  self.database_name,
                                                  self.continue_from_existing_database)

            self.logger.info(
                f"Optimization study '{self.study_name}' initialized.")

            # Set up the Optuna sampler for suggesting new designs
            seed = self.config["agent_training_configuration"]["random_seed"]
            if seed == 'None' or seed == 'none':
                seed = None

            num_startup_trials = self.config["parameter_optimization_config"]["optuna"]["num_startup_trials"]
            n_ei_candidates = self.config["parameter_optimization_config"]["optuna"]["num_ei_candidates"]
            multivariate = self.config["parameter_optimization_config"]["optuna"]["multivariate"]

            # We're using TPE (Tree-structured Parzen Estimator) as our sampler
            # It's good at exploring complex parameter spaces
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=num_startup_trials,
                n_ei_candidates=n_ei_candidates,
                multivariate=multivariate,
                seed=seed,
                constant_liar=True
            )

            self.study.sampler = sampler

            # Set up our GPU tracking - we'll distribute jobs across available GPUs
            if self.num_gpus > 0:
                self.allocations = {gpu_id: []
                                    for gpu_id in range(self.num_gpus)}
            else:
                self.allocations = {'cpu': []}  # No GPUs? Just use CPU

            # How many trials have we completed so far?
            trials_completed = self.get_completed_trials_count()

            # Main optimization loop - keep running batches until stopped
            while not self.stop_requested:
                # Figure out how many designs to run in this batch
                batch_size = self.batch_optim

                self.logger.info(f"Starting a new batch of size {batch_size}. "
                                 f"(Total started so far: {trials_completed})")

                # Update our batch status for the frontend
                self.current_batch_info = {
                    "batch_in_progress": True,
                    "trials": [],
                    "message": f"Batch of size {batch_size} started; not finished yet."
                }

                # Lists to track the processes and trials in this batch
                batch_processes = []
                batch_trialinfo = []

                # Launch each trial in the batch
                for _ in range(batch_size):
                    # Don't exceed our parallelism limit - wait if needed
                    while self.count_active_processes() >= self.num_parallel_evaluations:
                        await self.cleanup_finished_processes()
                        await asyncio.sleep(1)

                    # Ask Optuna for a new trial with suggested parameters
                    trial = self.study.ask()
                    param_values = self.sample_hyperparams_for_trial(trial)

                    # Find a GPU with available capacity
                    gpu_id = self.find_free_gpu_or_cpu()

                    # Start a process to evaluate this design
                    process, result_file_path = await self.start_evaluator(
                        trial,
                        param_values,
                        config_file_path,
                        gpu_id
                    )
                    # Keep track of where this process is running
                    self.allocations[gpu_id].append(process)
                    self.active_trials.append(process)
                    self.process_to_trialinfo[process.pid] = (
                        trial, result_file_path)

                    # Track for this batch specifically
                    batch_processes.append(process)
                    batch_trialinfo.append((trial, result_file_path))

                    trials_completed += 1
                    self.logger.info(
                        f"Started trial {trial.number}. "
                        f"Currently running: {self.count_active_processes()} processes."
                    )

                # Now we wait for all the batch processes to complete
                while not all(p.poll() is not None for p in batch_processes):
                    await self.cleanup_finished_processes()
                    await asyncio.sleep(1)

                # Final cleanup pass
                await self.cleanup_finished_processes()

                # Gather all the results from this batch
                batch_results = []
                for (trial, result_file_path) in batch_trialinfo:
                    simulation_value = self.get_objective_from_file(
                        result_file_path)

                    # Skip trials that failed (returned NaN)
                    if math.isnan(simulation_value):
                        self.logger.warning(
                            f"Trial {trial.number} returned NaN. Skipping trial.")
                        continue

                    entry = {
                        "trial": trial,
                        "simulation_value": simulation_value,
                        "result_path": result_file_path,
                    }

                    if self._uses("wave"):  # only when Wave is among the sources
                        wave_metric_value = self.get_wave_metric_from_file(
                            result_file_path)
                        entry["wave_metric"] = wave_metric_value
                        self.logger.info(
                            f"Trial {trial.number}: Calculated wave metric = {wave_metric_value}")

                    # Save this trial's data
                    batch_results.append(entry)

                # Prepare all the trial data to show to the user
                trials_data_for_json = []
                video_paths = {}  # Track video paths by trial number

                for br in batch_results:
                    trial_obj = br["trial"]
                    simulation_value = br["simulation_value"]

                    # Build video URL under run/<user_id>/<study>/results/...
                    if self.task_name == "chart_reader":
                        mode = self.chart_mode or "tool"
                        ctask = self.chart_task or "theme_parks"
                        video_rel_path = (
                            f"run/{self.user_id}/chartist_task/{mode}/{ctask}/results/"
                            f"trial_{trial_obj.number}/generated_charts/chart_design.mp4"
                        )
                    elif self.task_name == "toggle_task":
                        # toggle: no video
                        video_rel_path = None
                    else:
                        video_filename = f"trial_{trial_obj.number}_video.mp4"
                        video_rel_path = (
                            f"run/{self.user_id}/{self.study_name}/results/videos_saved/{video_filename}"
                        )

                    # Remember this video path
                    video_paths[trial_obj.number] = video_rel_path

                    # Save the simulation result to our dataset
                    self.record_trial(trial_obj.number,
                                      trial_obj.params, simulation_value)

                    if self._uses("wave") and br.get("wave_metric") is not None:
                        self.wave_metric_values[trial_obj.number] = br["wave_metric"]
                        self.logger.info(
                            f"Stored wave metric for trial {trial_obj.number}: {br['wave_metric']}")
                    elif self._uses("wave"):
                        self.logger.warning(
                            f"Wave source selected but no wave metric available for trial {trial_obj.number}")

                # -------------------------------------------------------
                # First GP fit - to show disagreement info to the user
                # This helps them make informed decisions when selecting designs
                # -------------------------------------------------------
                lambda_ = self.lambd
                r_dict, g_dict, f_dict, w_dict, disagreement_dict, uncertainty_dict = fit_all_gp(
                    self.trials_dataset,
                    self.user_preferences_dataset,
                    lambda_=lambda_,
                    param_bounds=self.param_bounds,
                    y_min=self.y_min,
                    y_max=self.y_max,
                    g_min=self.g_min,
                    g_max=self.g_max,
                    g_clip_eps=0.02,
                    wave_metric_source=(
                        self.wave_metric_values if self._uses("wave") else None),
                    use_sources=self.sources_to_use,
                )

                # Save disagreement and uncertainty info
                self.disagreement_data = disagreement_dict
                self.uncertainty_data = uncertainty_dict

                # Save GP predictions for later use (e.g., in get_best_trial)
                self.r_predictions = r_dict  # LLM predictions
                self.g_predictions = g_dict  # Human predictions
                self.w_predictions = w_dict  # Wave predictions

                # Add info about each trial to send to the frontend
                for br in batch_results:
                    trial_obj = br["trial"]
                    simulation_value = br["simulation_value"]

                    # Format the disagreement info for display
                    disagreement_display = None
                    # Only show disagreement if this isn't the first batch (need preferences for that)
                    if not self.user_preferences_dataset.empty and trial_obj.number in self.disagreement_data:
                        disagreement_value = self.disagreement_data[trial_obj.number]
                        uncertainty_value = self.uncertainty_data.get(
                            trial_obj.number, 0)

                        # Make it readable
                        disagreement_display = f"Predicted disagreement: {disagreement_value:.2f}±{uncertainty_value:.2f}"

                    if disagreement_display:
                        self.logger.info(
                            f"Trial {trial_obj.number} has disagreement: {disagreement_display}")
                    else:
                        self.logger.info(
                            f"Trial {trial_obj.number} - no disagreement data shown (first batch)")

                    # Get the video path we stored earlier
                    video_rel_path = video_paths[trial_obj.number]

                    # Basic trial data
                    trial_data = {
                        "trial_number": trial_obj.number,
                        "simulated_value": simulation_value,
                        "params": trial_obj.params,
                        "video_url": video_rel_path
                    }

                    # Add metric values ONLY for explicitly selected sources
                    self.logger.info(
                        f"Trial {trial_obj.number}: Adding metrics for sources {self.sources_to_use} (explicit: {self.sources_explicit})")

                    # Only show LLM metric if explicitly selected
                    if self.sources_explicit and self._uses("llm") and trial_obj.number in r_dict:
                        trial_data["llm_value"] = r_dict[trial_obj.number]
                        self.logger.info(
                            f"Trial {trial_obj.number}: Added LLM metric = {r_dict[trial_obj.number]}")

                    # Only show Wave metric if explicitly selected
                    if self.sources_explicit and self._uses("wave") and trial_obj.number in w_dict:
                        trial_data["wave_value"] = w_dict[trial_obj.number]
                        self.logger.info(
                            f"Trial {trial_obj.number}: Added Wave metric = {w_dict[trial_obj.number]}")
                    elif self.sources_explicit and self._uses("wave"):
                        self.logger.warning(
                            f"Trial {trial_obj.number}: Wave source selected but no wave metric available")

                    # Always add prediction error for uncertainty display
                    if trial_obj.number in uncertainty_dict:
                        trial_data["prediction_error"] = uncertainty_dict[trial_obj.number]

                    # Add disagreement data after the first batch
                    if not self.user_preferences_dataset.empty:
                        trial_data.update({
                            "disagreement_display": disagreement_display,
                            "disagreement_value": self.disagreement_data.get(trial_obj.number, 0),
                            "uncertainty_value": self.uncertainty_data.get(trial_obj.number, 0)
                        })

                    trials_data_for_json.append(trial_data)

                self.logger.info(
                    "\nBatch Completed. Below are the results for each trial in this batch:\n")

                # Log all the trial results nicely
                for i, br in enumerate(batch_results):
                    tr = br["trial"]
                    obj_val = br["simulation_value"]
                    self.logger.info(
                        f"   [Trial #{tr.number}] Objective: {obj_val}\n"
                        f"       Parameters: {tr.params}\n"
                    )

                # Let the frontend know the batch is ready for user review
                self.current_batch_info["batch_in_progress"] = False
                self.current_batch_info["trials"] = trials_data_for_json
                self.current_batch_info["message"] = "Batch finished. Waiting for user input."

                self.logger.info("Batch finished. Waiting for user input.")
                # Persist CSV/Parquet tables under datasets/
                self._flush_datasets()

                # Wait until the user tells us which designs they prefer
                while self.chosen_trials is None and not self.stop_requested:
                    await asyncio.sleep(1)

                if self.stop_requested:
                    self.logger.info(
                        "Stop requested. Exiting optimization loop.")
                    break

                # Get the user's selections
                chosen_trials = self.chosen_trials
                not_chosen_trials = self.not_chosen_trials

                # Reset for the next batch
                self.chosen_trials = None
                self.not_chosen_trials = None

                # Record user preferences
                pref_data = {
                    "chosen_trials": chosen_trials,
                    "not_chosen_trials": not_chosen_trials
                }

                # Add to our preference dataset
                if self.user_preferences_dataset.empty:
                    self.user_preferences_dataset = pd.DataFrame([pref_data])
                else:
                    self.user_preferences_dataset = pd.concat(
                        [self.user_preferences_dataset,
                            pd.DataFrame([pref_data])],
                        ignore_index=True
                    )

                # -------------------------------------------------------
                # Second GP fit - with the user's new preferences
                # This blends simulation data with what the user likes
                # -------------------------------------------------------
                lambda_ = self.lambd
                r_dict, g_dict, f_dict, w_dict, disagreement_dict, uncertainty_dict = fit_all_gp(
                    self.trials_dataset,
                    self.user_preferences_dataset,
                    lambda_=lambda_,
                    param_bounds=self.param_bounds,
                    y_min=self.y_min,
                    y_max=self.y_max,
                    g_min=self.g_min,
                    g_max=self.g_max,
                    g_clip_eps=0.02,
                    wave_metric_source=(
                        self.wave_metric_values if self._uses("wave") else None),
                    use_sources=self.sources_to_use,
                )

                # Update disagreement info with new data
                self.disagreement_data = disagreement_dict
                self.uncertainty_data = uncertainty_dict

                # Tell Optuna about our results, using the combined objective
                # from both simulation and user preferences
                for br in batch_results:
                    trial_obj = br["trial"]
                    tnum = trial_obj.number
                    # simulator GP posterior mean
                    sim_val = r_dict.get(tnum)
                    # human GP posterior mean
                    human_val = g_dict.get(tnum)
                    wave_val = self.wave_metric_values.get(
                        tnum) if self._uses("wave") else None
                    # combined objective
                    final_obj = f_dict.get(tnum)

                    # persist a full row for this trial
                    self.record_trial_full(
                        trial_number=tnum,
                        parameters=trial_obj.params,
                        simulated_value=sim_val,
                        human_value=human_val,
                        combined_value=final_obj,
                        wave_metric=wave_val,
                    )

                    # Update Optuna with the final objective
                    self.study.tell(trial_obj, final_obj)

                    self._log_trial_sources(
                        tnum,
                        r_dict=r_dict,
                        g_dict=g_dict,
                        f_val=final_obj,
                        disagreement_dict=disagreement_dict,
                        uncertainty_dict=uncertainty_dict,
                    )

                # write all datasets including the new one
                self._flush_datasets()

                # Clean up the batch processes from our allocation tracking
                self.clear_finished_from_allocations(batch_processes)

            self.logger.info(
                f"Optimization terminated after {trials_completed} trials.")

            return f"Optimization terminated after {trials_completed} trials completed for study: {self.study_name}."

    async def cleanup_finished_processes(self):
        """
        Housekeeping - find any processes that have finished and remove them
        from our active tracking lists.
        """
        for gpu_id, processes in list(self.allocations.items()):
            for process in processes[:]:
                if process.poll() is not None:  # Process has exited
                    processes.remove(process)
                    if process in self.active_trials:
                        self.active_trials.remove(process)

                    if gpu_id == 'cpu':
                        self.logger.info(
                            f"Process (pid={process.pid}) on CPU has finished.")
                    else:
                        self.logger.info(
                            f"Process (pid={process.pid}) on GPU-{gpu_id} has finished.")

    def clear_finished_from_allocations(self, processes_in_batch):
        """
        More housekeeping - after a batch is done and we've processed the results,
        clean up any remaining process references.
        """
        for gpu_id, processes in self.allocations.items():
            for p in processes_in_batch:
                if p in processes:
                    processes.remove(p)

    def count_active_processes(self):
        """Count how many processes are currently running across all GPUs/CPU."""
        return sum(len(proc_list) for proc_list in self.allocations.values())

    def find_free_gpu_or_cpu(self):
        """
        Find the least busy GPU (or CPU) to run our next process.
        Tries to distribute the load evenly.
        """
        if self.num_gpus > 0:
            # Pick the GPU with the fewest current processes
            gpu_id = min(self.allocations,
                         key=lambda g: len(self.allocations[g]))
            return gpu_id
        else:
            return 'cpu'  # No GPUs available

    def sample_hyperparams_for_trial(self, trial: optuna.Trial):
        """
        Create parameter values for this trial by sampling from the
        ranges defined in the config file.
        """
        param_bounds = self.config["parameters_bounds"]
        param_values = {}
        for param_name, config_ in param_bounds.items():
            if config_.get('searchable', False):
                if config_.get('integer', False):
                    # For integer parameters (like counts or iterations)
                    param_values[param_name] = trial.suggest_int(
                        name=param_name,
                        low=config_['start'],
                        high=config_['stop']
                    )
                else:
                    # For floating point parameters
                    log = config_.get('log', False)
                    param_values[param_name] = trial.suggest_float(
                        name=param_name,
                        low=config_['start'],
                        high=config_['stop'],
                        log=log  # Log scale is useful for parameters with large ranges
                    )
            else:
                # Use fixed values for parameters that shouldn't be optimized
                param_values[param_name] = config_['user_preference']
        return param_values

    async def start_evaluator(self, trial, param_values, config_file_path, gpu_id='cpu'):
        """Start a process to evaluate the given design parameters."""
        if self.task_name == "chart_reader":
            process, result_file_path = start_evaluator_chatreader(
                self.study_name, trial, param_values, config_file_path, gpu_id=gpu_id,
                chart_mode=(self.chart_mode or "tool"),
                chart_task=(self.chart_task or "theme_parks"),
            )
        elif self.task_name == "toggle_task":
            process, result_file_path = start_evaluator_toggle(
                self.study_name, trial, param_values, config_file_path,
                gpu_id=gpu_id, toggle_mode=self._toggle_mode_folder())
        else:
            process, result_file_path = start_evaluator_tutorial(
                self.study_name, trial, param_values, config_file_path, gpu_id=gpu_id)

        return process, result_file_path

    def get_objective_from_file(self, filepath):
        """
        Read the results file that an evaluator wrote when it finished.
        Returns NaN if anything went wrong (file missing, bad format, etc.)
        """
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                return float(data["objective_value"])
            else:
                return float("nan")
        except:
            return float("nan")

    def get_completed_trials_count(self):
        """How many trials have we finished so far?"""
        return len(self.study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))

    def get_or_create_study(self, study_name, database_name, continue_from_existing_database):
        """
        Set up our Optuna study - either create a new one or continue from
        an existing one, depending on user preference.
        """
        db_dir = self._paths()["db_dir"]
        os.makedirs(db_dir, exist_ok=True)
        database_url = f"sqlite:///{os.path.join(db_dir, database_name + '.db')}"

        try:
            if continue_from_existing_database:
                # Try to load the existing study
                self.logger.info(
                    f"Attempting to load study: {study_name} from database: {database_url}")
                study = optuna.load_study(
                    study_name=study_name, storage=database_url)
                self.logger.info(f"Study {study_name} found.")
            else:
                # Delete the old study if it exists
                try:
                    self.logger.info(
                        f"Checking for existing study: {study_name} in database: {database_url}")
                    optuna.delete_study(
                        study_name=study_name, storage=database_url)
                    self.logger.info(f"Existing study {study_name} deleted.")
                except KeyError:
                    self.logger.info(
                        f"Study {study_name} does not exist in the database. No deletion required.")

                # Create a fresh study
                self.logger.info(
                    f"Creating a new study: {study_name} in database: {database_url}")
                study = optuna.create_study(
                    study_name=study_name, storage=database_url, direction="maximize")
                self.logger.info(f"Study {study_name} created.")
        except Exception as e:
            if "Record does not exist" in str(e):
                # Study doesn't exist, so create it
                self.logger.info(
                    f"Study {study_name} not found. Creating a new one.")
                study = optuna.create_study(
                    study_name=study_name, storage=database_url, direction="maximize")
                self.logger.info(f"Study {study_name} created.")
            else:
                raise e  # Some other error occurred

        return study

    def get_best_trial(self):
        """
        Find the best design we've discovered so far, with all
        relevant details to display to the user.
        """
        try:
            if not self.study:
                message = "Optimization has not started yet."
                self.logger.info(message)
                return None, None, None, None, message

            if self.study.best_trial is None:
                message = "No completed trials available to determine the best trial."
                self.logger.info(message)
                return None, None, None, None, message

            # Get the best trial according to Optuna
            best_trial = self.study.best_trial
            self.logger.info(f"Best trial objective value: {best_trial.value}")
            self.logger.info(f"Best trial number: {best_trial.number}")
            self.logger.info("Parameters:")
            for key, value in best_trial.params.items():
                self.logger.info(f"    {key}: {value}")

            # Get the simulation value for this trial
            matching_rows = self.trials_dataset[self.trials_dataset['trial_number']
                                                == best_trial.number]
            if not matching_rows.empty:
                simulated_based_value = matching_rows.iloc[0]['simulated_based_value']
            else:
                simulated_based_value = None

            # Get metric values based on selected sources
            llm_value = None
            human_value = None
            wave_value = None
            prediction_error = None

            self.logger.info(
                f"Best trial {best_trial.number}: Determining metrics for sources {self.sources_to_use} (explicit: {self.sources_explicit})")

            # Only show LLM metric if explicitly selected
            if self.sources_explicit and hasattr(self, 'r_predictions') and self._uses("llm") and best_trial.number in self.r_predictions:
                llm_value = self.r_predictions[best_trial.number]
                self.logger.info(f"Best trial: LLM metric = {llm_value}")
            elif self.sources_explicit and self._uses("llm"):
                self.logger.warning(
                    f"Best trial: LLM source selected but no LLM prediction available")

            # Never show human metric per user request
            # if hasattr(self, 'g_predictions') and self._uses("human") and best_trial.number in self.g_predictions:
            #     human_value = self.g_predictions[best_trial.number]

            # Only show Wave metric if explicitly selected
            if self.sources_explicit and hasattr(self, 'w_predictions') and self._uses("wave") and best_trial.number in self.w_predictions:
                wave_value = self.w_predictions[best_trial.number]
                self.logger.info(f"Best trial: Wave metric = {wave_value}")
            elif self.sources_explicit and self._uses("wave"):
                self.logger.warning(
                    f"Best trial: Wave source selected but no wave prediction available")

            # Always show prediction error
            if hasattr(self, 'uncertainty_data') and best_trial.number in self.uncertainty_data:
                prediction_error = self.uncertainty_data[best_trial.number]

            # Add disagreement info for the frontend
            disagreement_data = None
            if best_trial.number in self.disagreement_data:
                disagreement_value = self.disagreement_data[best_trial.number]
                uncertainty_value = self.uncertainty_data.get(
                    best_trial.number, 0)

                disagreement_data = {
                    "value": disagreement_value,
                    "uncertainty": uncertainty_value,
                    "display": f"Predicted disagreement: {disagreement_value:.2f}±{uncertainty_value:.2f}",
                    "llm_value": llm_value,
                    "human_value": human_value,
                    "wave_value": wave_value,
                    "prediction_error": prediction_error
                }

            # Build best video relative path (same rules as in batch construction)
            if self.task_name == "chart_reader":
                mode = self.chart_mode or self.config.get("configuration", {}).get("chart_mode") or "tool"
                ctask = self.chart_task or self.config.get("configuration", {}).get("chart_task") or "theme_parks"
                video_rel_path = (
                    f"run/{self.user_id}/chartist_task/{mode}/{ctask}/results/"
                    f"trial_{best_trial.number}/generated_charts/chart_design.mp4"
                )
            elif self.task_name == "toggle_task":
                video_rel_path = None
            else:
                # tutorial (and other RL sims)
                video_rel_path = (
                    f"run/{self.user_id}/{self.study_name}/results/videos_saved/trial_{best_trial.number}_video.mp4"
                )

            payload = {
                "best_value": best_trial.value,
                "best_params": best_trial.params,
                "best_trial_number": best_trial.number,
                "simulated_based_value": simulated_based_value,
                "disagreement_data": disagreement_data,
                "video_url": video_rel_path,
            }
            return payload

        except Exception as e:
            if str(e) == "Record does not exist.":
                message = "No completed trials available to determine the best trial."
                self.logger.info(message)
                return {
                    "best_value": None,
                    "best_params": None,
                    "best_trial_number": None,
                    "simulated_based_value": None,
                    "disagreement_data": None,
                    "video_url": None,
                    "message": message,
                }

            self.logger.error(f"An unexpected error occurred: {e}")
            return {
                "best_value": None,
                "best_params": None,
                "best_trial_number": None,
                "simulated_based_value": None,
                "disagreement_data": None,
                "video_url": None,
                "message": str(e),
            }

    def get_optimization_status(self):
        """Are we currently running? How many trials have we completed?"""
        completed_trials = self.get_completed_trials_count()
        status = "stopped" if self.stop_requested else "running"

        return status, completed_trials

    def record_trial(self, trial_number, parameters, simulation_value):
        """
        Save information about a completed trial to our dataset
        so we can use it for fitting the GP models.
        """
        trial_data = {
            "trial_number": trial_number,
            "parameters": parameters,
            "simulated_based_value": simulation_value
        }
        if self.trials_dataset.empty:
            self.trials_dataset = pd.DataFrame([trial_data])
        else:
            self.trials_dataset = pd.concat(
                [self.trials_dataset, pd.DataFrame([trial_data])],
                ignore_index=True
            )
        self.logger.info(
            f"Recorded trial {trial_number} with simulation-based value {simulation_value}.")

    def get_current_batch_info(self):
        """
        Get the status and results of the current batch of designs
        so we can display them to the user.
        """
        return self.current_batch_info

    async def stop_optimization(self):
        """
        Emergency stop! Halt all optimization, kill running processes,
        and snapshot best-trial artifacts into results/best_trial/.
        """
        self.stop_requested = True
        self.logger.info("Stop requested. Terminating all processes.")

        # Kill any processes that are still running
        for process in self.active_trials:
            if process.poll() is None:  # Still running
                self.logger.info(f"Terminating process (pid={process.pid}).")
                process.terminate()

        # Give them a moment to shut down gracefully
        await asyncio.sleep(1)

        # -------- Remove artifacts for trials not told to Optuna --------
        try:
            # Determine the highest trial number that has been told (COMPLETE)
            last_told = None
            if self.study:
                complete_trials = self.study.get_trials(
                    states=[optuna.trial.TrialState.COMPLETE]
                )
                if complete_trials:
                    last_told = max(t.number for t in complete_trials)

            p = self._paths()
            results_root = p["results"]
            os.makedirs(results_root, exist_ok=True)

            def _parse_trial_num_from_name(name: str):
                # matches trial_123 or trial_123_something
                if not name.startswith("trial_"):
                    return None
                try:
                    rest = name[len("trial_"):]
                    num_str = ""
                    for ch in rest:
                        if ch.isdigit():
                            num_str += ch
                        else:
                            break
                    return int(num_str) if num_str else None
                except Exception:
                    return None

            if self.task_name == "chart_reader":
                # results/trial_{N}/ directories
                if os.path.isdir(results_root):
                    for entry in os.listdir(results_root):
                        trial_num = _parse_trial_num_from_name(entry)
                        if trial_num is None:
                            continue
                        # If no told trials exist, delete all; else delete > last_told
                        if (last_told is None) or (trial_num > last_told):
                            target = os.path.join(results_root, entry)
                            try:
                                shutil.rmtree(target, ignore_errors=True)
                                self.logger.info(
                                    f"Removed artifacts for untold trial: {target}")
                            except Exception as e:
                                self.logger.debug(
                                    f"Failed to remove {target}: {e}")
            else:
                # tutorial & toggle_task:
                trials_dir = os.path.join(results_root, "trials")
                if os.path.isdir(trials_dir):
                    for fname in os.listdir(trials_dir):
                        tnum = _parse_trial_num_from_name(fname)
                        if tnum is None:
                            continue
                        if (last_told is None) or (tnum > last_told):
                            fpath = os.path.join(trials_dir, fname)
                            try:
                                os.remove(fpath)
                                self.logger.info(
                                    f"Removed artifact for untold trial: {fpath}")
                            except IsADirectoryError:
                                shutil.rmtree(fpath, ignore_errors=True)
                                self.logger.info(
                                    f"Removed folder for untold trial: {fpath}")
                            except Exception as e:
                                self.logger.debug(
                                    f"Failed to remove {fpath}: {e}")

                # Also clean videos (if any)
                videos_dir = os.path.join(results_root, "videos_saved")
                if os.path.isdir(videos_dir):
                    for fname in os.listdir(videos_dir):
                        tnum = _parse_trial_num_from_name(fname)
                        if tnum is None:
                            continue
                        if (last_told is None) or (tnum > last_told):
                            fpath = os.path.join(videos_dir, fname)
                            try:
                                os.remove(fpath)
                                self.logger.info(
                                    f"Removed video for untold trial: {fpath}")
                            except Exception as e:
                                self.logger.debug(
                                    f"Failed to remove {fpath}: {e}")
        except Exception as e:
            self.logger.exception(f"Failed while cleaning untold trials: {e}")

        # -------- Copy best trial artifacts --------
        try:
            if not self.study or self.study.best_trial is None:
                self.logger.info(
                    "No completed trials available; skipping best-trial snapshot.")
                return

            best_trial = self.study.best_trial
            best_num = best_trial.number
            trial_id = f"trial_{best_num}"

            p = self._paths()
            results_root = p["results"]
            os.makedirs(results_root, exist_ok=True)

            best_out_dir = os.path.join(results_root, "best_trial")
            shutil.rmtree(best_out_dir, ignore_errors=True)
            os.makedirs(best_out_dir, exist_ok=True)

            if self.task_name == "chart_reader":
                # For chart_reader, each trial has its own directory: results/trial_{N}/...
                trial_src_dir = os.path.join(results_root, trial_id)
                if os.path.isdir(trial_src_dir):
                    # Copy the whole trial folder inside best_trial/
                    shutil.copytree(trial_src_dir, os.path.join(
                        best_out_dir, trial_id))
                    self.logger.info(
                        f"Copied best chart_reader trial folder: {trial_src_dir} -> {best_out_dir}")
                else:
                    self.logger.info(
                        f"Expected chart_reader trial dir not found: {trial_src_dir}")
            else:
                # For tutorial & toggle_task, copy matching files from results/trials/
                trials_dir = os.path.join(results_root, "trials")
                if os.path.isdir(trials_dir):
                    for name in os.listdir(trials_dir):
                        if name.startswith(f"{trial_id}_"):
                            src = os.path.join(trials_dir, name)
                            dst = os.path.join(best_out_dir, name)
                            try:
                                shutil.copy2(src, dst)
                            except Exception as e:
                                self.logger.debug(f"Skip copying {src}: {e}")
                    self.logger.info(
                        f"Copied trial-matched files from {trials_dir} to {best_out_dir}")

                # Also copy any video (tutorial may write to results/videos_saved/)
                videos_dir = os.path.join(results_root, "videos_saved")
                if os.path.isdir(videos_dir):
                    for name in os.listdir(videos_dir):
                        # videos can be named like trial_{N}_video.mp4
                        if name.startswith(f"{trial_id}_") or name.startswith(trial_id):
                            src = os.path.join(videos_dir, name)
                            dst = os.path.join(best_out_dir, name)
                            try:
                                shutil.copy2(src, dst)
                            except Exception as e:
                                self.logger.debug(f"Skip copying {src}: {e}")
                    self.logger.info(
                        f"Copied any matching videos from {videos_dir} to {best_out_dir}")

            self.logger.info(
                f"Best trial #{best_num} artifacts saved to: {best_out_dir}")
        except Exception as e:
            self.logger.exception(
                f"Failed to snapshot best-trial artifacts: {e}")


def start_evaluator_tutorial(study_name, trial, param_values, config_file_path, gpu_id='cpu'):
    """
    Start an evaluator process for the tutorial task.
    This runs a simulation to evaluate how good a design is.
    """
    env = os.environ.copy()
    if gpu_id != 'cpu':
        # Tell the process which GPU to use
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        # Don't use any GPU
        env["CUDA_VISIBLE_DEVICES"] = ""

    # Suppress warning messages
    env["PYTHONWARNINGS"] = "ignore"

    # Write the parameters to a file the evaluator can read
    current_folder = os.path.abspath(os.getcwd())
    uid = os.environ.get("PARTICIPANT_ID", "anon")
    results_dir = os.path.join(
        current_folder, "run", uid, study_name, "results", "trials")
    os.makedirs(results_dir, exist_ok=True)

    trial_id = f"trial_{trial.number}"
    hyperparam_file_path = os.path.join(
        results_dir, f"{trial_id}_hyperparams.json")
    with open(hyperparam_file_path, 'w') as hp_file:
        json.dump(param_values, hp_file, indent=2)

    env["HYPERPARAMS_FILE"] = hyperparam_file_path

    # Set up the result file where the evaluator will write its output
    result_file_path = os.path.join(results_dir, f"{trial_id}_result.json")
    env["TRIAL_ID"] = trial_id
    env["RESULT_FILE"] = result_file_path
    env["PARTICIPANT_ID"] = uid

    # Start the evaluator process
    process = Popen(
        ["python", "evaluator.py", config_file_path],
        env=env,
        stdout=PIPE,
        stderr=PIPE
    )
    return process, result_file_path


# orchestrator.py


def start_evaluator_chatreader(
    study_name, trial, param_values, config_file_path, gpu_id='cpu',
    chart_mode='tool', chart_task='theme_parks'
):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu_id == 'cpu' else str(gpu_id)
    env["PYTHONWARNINGS"] = "ignore"

    trial_id = f"trial_{trial.number}"
    current_folder = os.path.abspath(os.getcwd())
    uid = os.environ.get("PARTICIPANT_ID", "anon")

    results_dir = os.path.join(
        current_folder, "run", uid, "chartist_task", chart_mode, chart_task,
        "results", trial_id
    )
    os.makedirs(results_dir, exist_ok=True)

    # --- pass hyperparams/result paths ---
    hyperparam_file_path = os.path.join(
        results_dir, f"{trial_id}_hyperparams.json")
    with open(hyperparam_file_path, 'w') as hp_file:
        json.dump(param_values, hp_file, indent=2)
    env["HYPERPARAMS_FILE"] = hyperparam_file_path

    result_file_path = os.path.join(results_dir, f"{trial_id}_result.json")
    env["TRIAL_ID"] = trial_id
    env["RESULT_FILE"] = result_file_path
    env["PARTICIPANT_ID"] = uid
    env["CHART_MODE"] = chart_mode
    env["CHART_TASK"] = chart_task

    # --- make chartist_case importable in the child ---
    this_dir = os.path.dirname(os.path.abspath(__file__))
    chartist_dir = os.path.join(this_dir, "chartist_case")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [
        this_dir,           # so 'chartist_case' is importable
        chartist_dir,       # so bare 'model' resolves to chartist_case/model
        env.get("PYTHONPATH")
    ]))

    # resolve script
    script_path = os.path.join(this_dir, "evaluator_chartist.py")

    # log stdout/stderr to files (no PIPE blocking)
    stdout_log = open(os.path.join(results_dir, "stdout.log"), "wb")
    stderr_log = open(os.path.join(results_dir, "stderr.log"), "wb")

    env.setdefault("SDL_VIDEODRIVER", "dummy")
    env.setdefault("SDL_AUDIODRIVER", "dummy")
    env.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    env.setdefault("MPLBACKEND", "Agg")

    proc = subprocess.Popen(
        [sys.executable, script_path, config_file_path],
        env=env, cwd=this_dir, stdout=stdout_log, stderr=stderr_log, close_fds=True
    )
    return proc, result_file_path


def start_evaluator_toggle(study_name, trial, param_values, config_file_path, gpu_id='cpu', toggle_mode='tool'):
    """
    Start an evaluator process for the toggle design task.
    This evaluation is specific to UI toggle designs without RL training.
    """
    import json
    import subprocess
    import traceback
    from subprocess import Popen, PIPE
    from toggle_renderer import save_toggle_png
    from third_party_modules.wave_metric.wave_metric import Metric
    import base64

    # Create results directory
    current_folder = os.path.abspath(os.getcwd())
    uid = os.environ.get("PARTICIPANT_ID", "anon")
    results_dir = os.path.join(
        current_folder, "run", uid, "toggle_task", (toggle_mode or "tool"), "results", "trials")
    os.makedirs(results_dir, exist_ok=True)

    # Create trial ID
    trial_id = f"trial_{trial.number}"

    # Save the parameters to a JSON file
    hyperparam_file_path = os.path.join(
        results_dir, f"{trial_id}_hyperparams.json")
    with open(hyperparam_file_path, 'w') as hp_file:
        json.dump(param_values, hp_file, indent=2)

    # Create result file path
    result_file_path = os.path.join(results_dir, f"{trial_id}_result.json")

    # For now, create a simple toggle evaluator that generates scores based on design parameters
    # This bypasses all the RL complexity and just evaluates toggle designs directly
    toggle_score = evaluate_toggle_design(param_values)

    # Render PNG
    png_path = os.path.join(results_dir, f"{trial_id}_toggle.png")

    try:
        save_toggle_png(param_values, png_path, is_on=True, size=140)
    except Exception:
        # write a per-trial error log so you can see *why* it failed
        err_path = os.path.join(results_dir, f"{trial_id}_png_error.txt")
        with open(err_path, "w") as ef:
            ef.write(traceback.format_exc())
        png_path = None

    # Handle PNG and wave metric calculation
    # Use None to indicate missing/failed metrics instead of 0.0 so
    # that downstream code (GP fitting) can ignore missing values.
    if png_path and os.path.exists(png_path):
        try:
            with open(png_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            res = Metric.execute_metric(b64)
            # defensive: Metric.execute_metric should return a list-like
            wave_metric = float(res[0]) if (
                res and res[0] is not None) else None
        except Exception:
            wave_metric = None
    else:
        wave_metric = None

    # Write the result directly (no subprocess needed for simple evaluation)
    result_data = {
        "objective_value": toggle_score,
        "wave_metric": wave_metric,
        "evaluation_completed": True,
        "trial_id": trial_id,
        "parameters": param_values,
    }

    with open(result_file_path, 'w') as result_file:
        json.dump(result_data, result_file, indent=2)

    # Return a mock process that's already completed
    # We'll create a dummy process that immediately finishes
    process = Popen(["echo", "Toggle evaluation completed"],
                    stdout=PIPE, stderr=PIPE)

    return process, result_file_path
