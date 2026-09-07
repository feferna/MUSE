import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt

import optuna
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from optuna.trial import FrozenTrial, TrialState
from optuna.distributions import UniformDistribution

# Add these imports for suppressing warnings and progress bar
import warnings
from tqdm import tqdm

# ============================================================================
# 0. SUPPRESS WARNINGS AND OPTUNA LOGGING
# ============================================================================
# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Only show warnings and errors
# Alternative: optuna.logging.disable_default_handler() to completely disable

# Suppress sklearn warnings specifically if needed
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================================================================
# 1. REPRODUCIBILITY SETUP
# ============================================================================
MASTER_SEED = 123  # Change this to get different but reproducible results

KERNEL = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)

def set_all_seeds(seed):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)

# ============================================================================
# 2. IMPORTS & GLOBAL CONFIGURATION
# ============================================================================
N_TRIALS = 50
phase_trial_count = N_TRIALS // 4  

# ============================================================================
# 3. ROSENBROCK-RELATED FUNCTIONS (used only as a reference definition)
# ============================================================================
def rosenbrock_2d(x0, x1):
    return (1 - x0)**2 + 100 * (x1 - x0**2)**2

# ============================================================================
# 4. RANDOM TRANSLATIONS FOR REALLY UNCORRELATED SOURCES
# ============================================================================
# Initialize with master seed for consistent shifts across all runs
np.random.seed(MASTER_SEED)
shift_s1 = 4.0 * np.random.rand(2) - 2.0
shift_s2 = 4.0 * np.random.rand(2) - 2.0
shift_s3 = 4.0 * np.random.rand(2) - 2.0

# ============================================================================
# 5. UNDERLYING (NOISE-FREE) FUNCTION DEFINITIONS
# ============================================================================
def underlying_rosenbrock(x0, x1):
    return (1 - x0)**2 + 100.0 * (x1 - x0**2)**2

def underlying_threehump(x0, x1):
    return 2*x0**2 - 1.05*x0**4 + (x0**6)/6 + x0*x1 + x1**2

def underlying_booth(x0, x1):
    return (x0 + 2*x1 - 7)**2 + (2*x0 + x1 - 5)**2

# ============================================================================
# 6. SOURCE DEFINITIONS WITH UNCORRELATED FUNCTIONS & RANDOM TRANSLATION
# ============================================================================
# We need to create noise generators that can be seeded per run
class SeededNoiseSource:
    def __init__(self, base_seed, source_id):
        self.base_seed = base_seed
        self.source_id = source_id
        self.call_count = 0
    
    def get_noise(self, noise_std):
        # Create a unique seed for each call
        seed = self.base_seed + self.source_id * 10000 + self.call_count
        rng = np.random.RandomState(seed)
        self.call_count += 1
        return noise_std * rng.randn()

# Global noise sources (will be reinitialized per run)
noise_s1 = None
noise_s2 = None
noise_s3 = None

def source1(x0, x1):
    return underlying_rosenbrock(x0 - shift_s1[0], x1 - shift_s1[1]) + noise_s1.get_noise(0.15)

def source2(x0, x1):
    return underlying_threehump(x0 - shift_s2[0], x1 - shift_s2[1]) + noise_s2.get_noise(0.10)

def source3(x0, x1):
    return underlying_booth(x0 - shift_s3[0], x1 - shift_s3[1]) + noise_s3.get_noise(0.05)

# ============================================================================
# 7. TRUE FUNCTION (for reference and plotting)
# ============================================================================
def true_function(x0, x1):
    val1 = underlying_rosenbrock(x0 - shift_s1[0], x1 - shift_s1[1])
    val2 = underlying_threehump(x0 - shift_s2[0], x1 - shift_s2[1])
    val3 = underlying_booth(x0 - shift_s3[0], x1 - shift_s3[1])
    return (val1 + val2 + val3) / 3.0

# ============================================================================
# 8. PHASE-SPECIFIC SOURCE & GP AVAILABILITY
# ============================================================================
def get_available_sources_phase(phase):
    if phase == 1:
        return [source1]
    elif phase == 2:
        return [source1, source2]
    elif phase == 3:
        return [source1, source2, source3]
    elif phase == 4:
        return [source1, source3]
    else:
        raise ValueError("Phase must be 1, 2, 3, or 4.")

def get_available_gps_phase(phase):
    if phase == 1:
        return [gp_s1]
    elif phase == 2:
        return [gp_s1, gp_s2]
    elif phase == 3:
        return [gp_s1, gp_s2, gp_s3]
    elif phase == 4:
        return [gp_s1, gp_s2, gp_s3]
    else:
        raise ValueError("Phase must be 1, 2, 3, or 4.")

# ============================================================================
# 9. GLOBAL GP KERNEL SETUP (used for dynamic surrogate)
# ============================================================================


# Global GPs (will be reinitialized per run)
gp_s1 = None
gp_s2 = None
gp_s3 = None

# ============================================================================
# 10. AGGREGATOR & OBJECTIVE FUNCTIONS
# ============================================================================
def aggregator_dynamic(x0, x1, gp_list):
    X_in = np.array([[x0, x1]])
    valid_preds = []
    for gp in gp_list:
        if hasattr(gp, "X_train_") and gp.X_train_.shape[0] > 0:
            pred = gp.predict(X_in, return_std=False)
            valid_preds.append(pred)
    if len(valid_preds) == 0:
        return 0.0
    return float(np.mean(valid_preds))

def dynamic_objective(trial, phase):
    global X_s1, Y_s1, X_s2, Y_s2, X_s3, Y_s3

    x0 = trial.suggest_float("x0", -3.0, 3.0)
    x1 = trial.suggest_float("x1", -3.0, 3.0)
    
    sources = get_available_sources_phase(phase)
    for func in sources:
        y_measured = func(x0, x1)
        if func is source1:
            X_s1 = np.vstack([X_s1, [[x0, x1]]])
            Y_s1 = np.append(Y_s1, y_measured)
        elif func is source2:
            X_s2 = np.vstack([X_s2, [[x0, x1]]])
            Y_s2 = np.append(Y_s2, y_measured)
        elif func is source3:
            X_s3 = np.vstack([X_s3, [[x0, x1]]])
            Y_s3 = np.append(Y_s3, y_measured)
    
    if X_s1.shape[0] > 0:
        gp_s1.fit(X_s1, Y_s1)
    if X_s2.shape[0] > 0:
        gp_s2.fit(X_s2, Y_s2)
    if X_s3.shape[0] > 0:
        gp_s3.fit(X_s3, Y_s3)
    
    gp_list = get_available_gps_phase(phase)
    pred_val = aggregator_dynamic(x0, x1, gp_list)
    return pred_val

def baseline_objective(trial, phase):
    x0 = trial.suggest_float("x0", -3.0, 3.0)
    x1 = trial.suggest_float("x1", -3.0, 3.0)
    sources = get_available_sources_phase(phase)
    evaluations = [func(x0, x1) for func in sources]
    return np.mean(evaluations)

# ============================================================================
# 11. FUNCTIONS TO RUN ONE COMPLETE OPTIMIZATION RUN (PHASES 1-4)
# ============================================================================
def run_dynamic_surrogate(run_seed):
    global X_s1, Y_s1, X_s2, Y_s2, X_s3, Y_s3, gp_s1, gp_s2, gp_s3
    global noise_s1, noise_s2, noise_s3

    # Initialize noise sources for this run
    noise_s1 = SeededNoiseSource(run_seed, 1)
    noise_s2 = SeededNoiseSource(run_seed, 2) 
    noise_s3 = SeededNoiseSource(run_seed, 3)

    # Reinitialize cumulative training data
    X_s1, Y_s1 = np.empty((0, 2)), np.empty((0,))
    X_s2, Y_s2 = np.empty((0, 2)), np.empty((0,))
    X_s3, Y_s3 = np.empty((0, 2)), np.empty((0,))
    
    # Initialize GP surrogates with seeded random states
    gp_s1 = GaussianProcessRegressor(kernel=KERNEL, alpha=0.15**2, normalize_y=False, random_state=run_seed + 1)
    gp_s2 = GaussianProcessRegressor(kernel=KERNEL, alpha=0.10**2, normalize_y=False, random_state=run_seed + 2)
    gp_s3 = GaussianProcessRegressor(kernel=KERNEL, alpha=0.05**2, normalize_y=False, random_state=run_seed + 3)
    
    all_candidates_dyn = []
    best_values = []      # Best (lowest) true value so far at end of each phase.
    phase_averages = []   # Average true value for each phase's candidates.

    for phase in [1, 2, 3, 4]:
        if X_s1.shape[0] > 0:
            gp_s1.fit(X_s1, Y_s1)
        if X_s2.shape[0] > 0:
            gp_s2.fit(X_s2, Y_s2)
        if X_s3.shape[0] > 0:
            gp_s3.fit(X_s3, Y_s3)
        
        # Re-evaluate previous candidates using the current GP surrogates.
        for cand in all_candidates_dyn:
            cand["value"] = aggregator_dynamic(cand["x0"], cand["x1"], get_available_gps_phase(phase))
        
        # Create study with seeded sampler
        sampler = optuna.samplers.TPESampler(seed=run_seed + phase * 100)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        
        # Seed study with all past candidates.
        for cand in all_candidates_dyn:
            now = datetime.now()
            frozen_trial = FrozenTrial(
                number=-1,
                trial_id=-1,
                state=TrialState.COMPLETE,
                value=cand["value"],
                params={"x0": cand["x0"], "x1": cand["x1"]},
                distributions={"x0": UniformDistribution(-3.0, 3.0),
                               "x1": UniformDistribution(-3.0, 3.0)},
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                datetime_start=now,
                datetime_complete=now,
            )
            study.add_trial(frozen_trial)
            
        phase_candidates = []
        
        def record_trial_callback(study, trial):
            phase_candidates.append({
                "x0": trial.params["x0"],
                "x1": trial.params["x1"],
                "value": trial.value,
                "time": trial.datetime_complete
            })
        
        # Optimize with progress bar disabled for Optuna
        study.optimize(lambda trial: dynamic_objective(trial, phase),
                       n_trials=phase_trial_count,
                       callbacks=[record_trial_callback],
                       show_progress_bar=False)  # Disable Optuna's progress bar
        
        phase_candidates.sort(key=lambda d: d["time"])
        all_candidates_dyn.extend(phase_candidates)
        
        # Compute best value over all candidates so far.
        current_best = np.inf
        for cand in all_candidates_dyn:
            val_true = true_function(cand["x0"], cand["x1"])
            if val_true < current_best:
                current_best = val_true
        best_values.append(current_best)
        
        # Compute average true value for this phase's candidates.
        phase_true_values = [true_function(cand["x0"], cand["x1"]) for cand in phase_candidates]
        phase_avg = np.mean(phase_true_values)
        phase_averages.append(phase_avg)
    
    return phase_averages, best_values

def run_baseline_retains(run_seed):
    global noise_s1, noise_s2, noise_s3
    
    # Initialize noise sources for this run (same as dynamic)
    noise_s1 = SeededNoiseSource(run_seed, 1)
    noise_s2 = SeededNoiseSource(run_seed, 2)
    noise_s3 = SeededNoiseSource(run_seed, 3)
    
    baseline_all_candidates = []
    best_values = []
    phase_averages = []
    
    for phase in [1, 2, 3, 4]:
        # Create study with seeded sampler
        sampler = optuna.samplers.TPESampler(seed=run_seed + phase * 100)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        
        phase_candidates = []
        
        def baseline_record_trial_callback(study, trial):
            phase_candidates.append({
                "x0": trial.params["x0"],
                "x1": trial.params["x1"],
                "value": trial.value,
                "time": trial.datetime_complete
            })
            
        # Optimize with progress bar disabled for Optuna
        study.optimize(lambda trial: baseline_objective(trial, phase), 
                       n_trials=phase_trial_count,
                       callbacks=[baseline_record_trial_callback],
                       show_progress_bar=False)  # Disable Optuna's progress bar
        
        phase_candidates.sort(key=lambda d: d["time"])
        baseline_all_candidates.extend(phase_candidates)
        
        current_best = np.inf
        for cand in baseline_all_candidates:
            val_true = true_function(cand["x0"], cand["x1"])
            if val_true < current_best:
                current_best = val_true
        best_values.append(current_best)
        
        phase_true_values = [true_function(cand["x0"], cand["x1"]) for cand in phase_candidates]
        phase_avg = np.mean(phase_true_values)
        phase_averages.append(phase_avg)
        
    return phase_averages, best_values

def run_baseline_discard(run_seed):
    global noise_s1, noise_s2, noise_s3
    
    # Initialize noise sources (same as before)
    noise_s1 = SeededNoiseSource(run_seed, 1)
    noise_s2 = SeededNoiseSource(run_seed, 2)
    noise_s3 = SeededNoiseSource(run_seed, 3)
    
    best_values = []
    phase_averages = []

    for phase in [1, 2, 3, 4]:
        # Restart optimization completely
        sampler = optuna.samplers.TPESampler(seed=run_seed + phase * 100)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        phase_candidates = []

        def record_trial_callback(study, trial):
            phase_candidates.append({
                "x0": trial.params["x0"],
                "x1": trial.params["x1"],
                "value": trial.value,
                "time": trial.datetime_complete
            })

        study.optimize(lambda trial: baseline_objective(trial, phase),
                       n_trials=phase_trial_count,
                       callbacks=[record_trial_callback],
                       show_progress_bar=False)

        phase_candidates.sort(key=lambda d: d["time"])

        # Compute best true value *just from this phase*
        current_best = min(true_function(c["x0"], c["x1"]) for c in phase_candidates)
        best_values.append(current_best)

        # Compute average true value for this phase
        avg_value = np.mean([true_function(c["x0"], c["x1"]) for c in phase_candidates])
        phase_averages.append(avg_value)

    return phase_averages, best_values

# ============================================================================
# 12. REPEAT THE EXPERIMENT 100 TIMES AND COLLECT RESULTS WITH PROGRESS BAR
# ============================================================================
n_runs = 100
dynamic_avg_results = []   # Dynamic surrogate: average per phase.
dynamic_best_results = []  # Dynamic surrogate: best value so far.
baseline_avg_results = []  # Baseline: average per phase.
baseline_best_results = [] # Baseline: best value so far.
baseline_discard_avg_results = []  # Baseline with restart: average per phase
baseline_discard_best_results = []

# Add progress bar for the main experiment loop
print(f"Running {n_runs} experiments with master seed: {MASTER_SEED}")
for run in tqdm(range(n_runs), desc="Experiment Progress", unit="run"):
    # Create unique but reproducible seed for each run
    run_seed = MASTER_SEED + run * 1000
    
    dyn_avg, dyn_best = run_dynamic_surrogate(run_seed)
    base_avg, base_best = run_baseline_retains(run_seed)
    discard_avg, discard_best = run_baseline_discard(run_seed)
    
    dynamic_avg_results.append(dyn_avg)
    dynamic_best_results.append(dyn_best)
    baseline_avg_results.append(base_avg)
    baseline_best_results.append(base_best)
    baseline_discard_avg_results.append(discard_avg)
    baseline_discard_best_results.append(discard_best)  

# Convert to NumPy arrays (each shape: (n_runs, 4)).
dynamic_avg_results = np.array(dynamic_avg_results)
dynamic_best_results = np.array(dynamic_best_results)
baseline_avg_results = np.array(baseline_avg_results)
baseline_best_results = np.array(baseline_best_results)
baseline_discard_avg_results = np.array(baseline_discard_avg_results)
baseline_discard_best_results = np.array(baseline_discard_best_results)  

# ============================================================================
# 13. COMPUTE GROUND TRUTH USING GRID SEARCH
# ============================================================================
print("Computing ground truth...")
grid_points = 100
x0_grid = np.linspace(-3, 3, grid_points)
x1_grid = np.linspace(-3, 3, grid_points)
ground_truth_value = np.inf

# Add progress bar for ground truth computation
for x0 in tqdm(x0_grid, desc="Ground truth computation", leave=False):
    for x1 in x1_grid:
        avg_eval = true_function(x0, x1)
        if avg_eval < ground_truth_value:
            ground_truth_value = avg_eval

# ============================================================================
# 14. PLOTTING BOXPLOTS FOR AVERAGE AND BEST RESULTS PER PHASE
# ============================================================================
import matplotlib.patches as mpatches

print("Generating plots...")
phases = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
# For grouped boxplots, we will create two positions for each phase.
positions_dyn     = [i - 0.3 for i in range(1, 5)]
positions_base    = [i       for i in range(1, 5)]
positions_discard = [i + 0.3 for i in range(1, 5)]
box_width = 0.22

# Prepare the data: for each phase, collect the results across all runs.
dyn_avg_data = [dynamic_avg_results[:, i] for i in range(4)]
base_avg_data = [baseline_avg_results[:, i] for i in range(4)]
dyn_best_data = [dynamic_best_results[:, i] for i in range(4)]
base_best_data = [baseline_best_results[:, i] for i in range(4)]
discard_avg_data = [np.array(baseline_discard_avg_results)[:, i] for i in range(4)]
discard_best_data = [np.array(baseline_discard_best_results)[:, i] for i in range(4)]

fig, axs = plt.subplots(2, 1, figsize=(14, 10))

# --- Subplot 1: Average True Objective Value ---
axs[0].boxplot(dyn_avg_data, positions=positions_dyn, widths=box_width, patch_artist=True,
               boxprops=dict(facecolor="lightblue"), medianprops=dict(color="blue"), showfliers=False)
axs[0].boxplot(base_avg_data, positions=positions_base, widths=box_width, patch_artist=True,
               boxprops=dict(facecolor="lightgreen"), medianprops=dict(color="green"), showfliers=False)
axs[0].boxplot(discard_avg_data, positions=positions_discard, widths=box_width, patch_artist=True,
               boxprops=dict(facecolor="lightgray"), medianprops=dict(color="black"), showfliers=False)

axs[0].axhline(y=ground_truth_value, color='red', linestyle='--', label="Ground Truth")
axs[0].set_title("Average True Objective Value per Phase (100 Runs)")
axs[0].set_ylabel("Average True Objective Value")
axs[0].set_xticks(range(1, 5))
axs[0].set_xticklabels(phases)
axs[0].grid(True, axis="y", linestyle="--")

legend_elements = [
    mpatches.Patch(facecolor="lightblue", edgecolor="blue", label="Multi-Surrogate (Ours)"),
    mpatches.Patch(facecolor="lightgreen", edgecolor="green", label="Single Surrogate (Retains Past Data)"),
    mpatches.Patch(facecolor="lightgray", edgecolor="black", label="Single Surrogate (Discards Past Data)"),
    mpatches.Patch(facecolor="red", edgecolor="red", label="Global Minimum")
]


axs[0].legend(handles=legend_elements, loc="best")

# --- Subplot 2: Best True Objective Value ---
axs[1].boxplot(dyn_best_data, positions=positions_dyn, widths=box_width, patch_artist=True,
               boxprops=dict(facecolor="lightblue"), medianprops=dict(color="blue"), showfliers=False)
axs[1].boxplot(base_best_data, positions=positions_base, widths=box_width, patch_artist=True,
               boxprops=dict(facecolor="lightgreen"), medianprops=dict(color="green"), showfliers=False)
axs[1].boxplot(discard_best_data, positions=positions_discard, widths=box_width, patch_artist=True,
               boxprops=dict(facecolor="lightgray"), medianprops=dict(color="black"), showfliers=False)

axs[1].axhline(y=ground_truth_value, color='red', linestyle='--', label="Ground Truth")
axs[1].set_title("Best True Objective Value per Phase (100 Runs)")
axs[1].set_ylabel("Best True Objective Value")
axs[1].set_xticks(range(1, 5))
axs[1].set_xticklabels(phases)
axs[1].grid(True, axis="y", linestyle="--")
axs[1].legend(handles=legend_elements, loc="best")

axs[1].set_xlabel("Phase")

plt.tight_layout()
plt.savefig("boxplots_average_and_best_solution_quality.svg")
plt.savefig("boxplots_average_and_best_solution_quality.pdf")
plt.close()

print("Boxplot saved as 'boxplots_average_and_best_solution_quality.svg'.")
print(f"Experiment completed with master seed: {MASTER_SEED}")
print("All results are now perfectly reproducible!")