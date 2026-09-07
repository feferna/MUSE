# utils_ns.py
import random
import numpy as np
import optuna
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.optimize import fmin_l_bfgs_b
import logging

class FixedMinMaxScaler:
    def __init__(self, low, high):
        self.low  = np.array(low, dtype=float)
        self.high = np.array(high, dtype=float)
        self.rng  = np.maximum(self.high - self.low, 1e-12)
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.low) / self.rng

# -----------------------------
# Core functions (Rosenbrock)
# -----------------------------
def rosenbrock(x):
    x1, x2 = x[0], x[1]
    return (1 - x1)**2 + 100.0 * (x2 - x1**2)**2

def rosenbrock_translated(x, shift):
    # Rosenbrock(x - shift)
    return rosenbrock(x - shift)

# -----------------------------
# Optimizer and GP utilities
# -----------------------------
def custom_optimizer(obj_func, initial_theta, bounds):
    x_opt, f_opt, _info = fmin_l_bfgs_b(
        func=obj_func,
        x0=initial_theta,
        bounds=bounds,
        maxiter=10000
    )
    return x_opt, f_opt

def fit_single_gp(X_data, y_vec, kernel=None, random_seed=None, x_scaler=None):
    if kernel is None:
        ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)
    if x_scaler is None:
        raise ValueError("Provide a fixed x_scaler.")

    X_scaled = x_scaler.transform(X_data)

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=20,
        optimizer=custom_optimizer,
        random_state=random_seed
    )
    gp.fit(X_scaled, y_vec.ravel())
    return gp

def fit_multi_gps(X_data, Y_data_list, kernel=None, random_seed=None, x_scaler=None):
    """
    Y_data_list: list of length M, each is shape (n_points,)
    Returns: list of (gp, X_scaler, Y_scaler)
    """
    models = []
    for y in Y_data_list:
        models.append(fit_single_gp(X_data, y, kernel=kernel, random_seed=random_seed, x_scaler=x_scaler))
    return models

def gp_predict(x, gp, x_scaler):
    x_scaled = x_scaler.transform(x.reshape(1, -1))
    y_mean, _ = gp.predict(x_scaled, return_std=True)
    
    return float(y_mean.ravel()[0])  # return as scalar

# -----------------------------
# Non-stationary shift schedule
# -----------------------------
def make_shift_schedule(step_size=10, delta_shift=np.array([0.5, 0.5]), start_shift=np.array([-1.0, -1.0])):
    """
    Returns a function s(t) that gives the cumulative translation at trial t.
    """
    step_size = int(step_size)
    delta_shift = np.array(delta_shift, dtype=float)
    start_shift = np.array(start_shift, dtype=float)

    def s(t):
        k = int(t // step_size)
        return start_shift + k * delta_shift
    return s

# -----------------------------
# Class: MSourceBayesOpt (Non-stationary)
# -----------------------------
class MSourceBayesOpt:
    """
    Study 5: Non-stationary, Rosenbrock-only.

    - True objective: F_true(x, t) = Rosenbrock(x - s(t))
    - M sources:      F_i(x, t)   = Rosenbrock(x - s(t) - b_i)

    - aggregation_mode = 'multi'  -> one GP per source; predict avg of GPs
    - aggregation_mode = 'single' -> one GP on the mean of sources at each x
    """
    def __init__(
        self,
        M,
        random_seed=42,
        x_range=[-2, 2],
        n_init=10,
        n_trials=50,
        shift_step_size=10,
        shift_per_step=(0.5, 0.5),
        start_shift=(-1.0, -1.0),
        source_biases=None,
        aggregation_mode="multi",  # 'multi' or 'single'
    ):
        assert aggregation_mode in ("multi", "single")
        self.M = int(M)
        self.random_seed = int(random_seed)
        
        self.x_range = x_range
        self._x_scaler = FixedMinMaxScaler(low=[self.x_range[0], self.x_range[0]],
                                   high=[self.x_range[1], self.x_range[1]])
        
        self.n_init = int(n_init)
        self.n_trials = int(n_trials)
        self.aggregation_mode = aggregation_mode

        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # Time index (trial counter); n_init happen first, then optimization trials
        self.current_t = 0

        # Define shift schedule s(t)
        self.s_t = make_shift_schedule(
            step_size=shift_step_size,
            delta_shift=np.array(shift_per_step, dtype=float),
            start_shift=np.array(start_shift, dtype=float),
        )

        # Define source-specific fixed biases b_i (M x 2)
        if source_biases is None:
            # Reasonable fixed biases around zero; reproducible
            rng = np.random.default_rng(self.random_seed + 999)
            self.biases = rng.uniform(low=-1.0, high=1.0, size=(self.M, 2))
        else:
            self.biases = np.array(source_biases, dtype=float)
            assert self.biases.shape == (self.M, 2)

        # Prepare data containers
        self.X_data = []
        if self.aggregation_mode == "multi":
            self.Y_data_list = [[] for _ in range(self.M)]  # per-source outputs
            self.Y_mean = None
        else:
            self.Y_mean = []  # mean of source outputs at each x
            self.Y_data_list = None

        # ---- Initial random evaluations (at t = 0..n_init-1)
        for _ in range(self.n_init):
            x1 = np.random.uniform(self.x_range[0], self.x_range[1])
            x2 = np.random.uniform(self.x_range[0], self.x_range[1])
            x_pt = np.array([x1, x2])
            self._observe_and_store(x_pt, t=self.current_t)
            self.current_t += 1  # advance time

        X_np = np.array(self.X_data)
        if self.aggregation_mode == "multi":
            Y_list_np = [np.array(y) for y in self.Y_data_list]
            self.gp_models = fit_multi_gps(X_np, Y_list_np, random_seed=self.random_seed, x_scaler=self._x_scaler)
        else:
            Y_mean_np = np.array(self.Y_mean)
            self.gp_model_single = fit_single_gp(X_np, Y_mean_np, random_seed=self.random_seed, x_scaler=self._x_scaler)

        # Optuna study (TPE proposes points, we return model-predicted value)
        logging.getLogger("optuna").setLevel(logging.WARNING)
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=self.random_seed,
                n_startup_trials=20,
                n_ei_candidates=20,
                multivariate=True
            )
        )

        # Metrics over time (per trial)
        self.trial_records = []  # list of dicts with x, t, f_true(x,t), dist_to_opt, incumbent_dist

    # ---- Non-stationary true function and sources ----
    def F_true(self, x, t):
        return rosenbrock_translated(x, self.s_t(t))

    def F_i(self, i, x, t):
        # source i Rosenbrock, biased by b_i
        return rosenbrock_translated(x, self.s_t(t) + self.biases[i])

    # ---- Internals ----
    def _observe_and_store(self, x, t):
        """Evaluate sources at (x,t) and store in the right buffers."""
        self.X_data.append(x)
        if self.aggregation_mode == "multi":
            for i in range(self.M):
                self.Y_data_list[i].append(self.F_i(i, x, t))
        else:
            vals = [self.F_i(i, x, t) for i in range(self.M)]
            self.Y_mean.append(np.mean(vals))

    def _predict_model(self, x):
        """Predict the model objective at x (no time in model)."""
        if self.aggregation_mode == "multi":
            preds = [gp_predict(x, self.gp_models[i], self._x_scaler) for i in range(self.M)]
            return float(np.mean(preds))
        else:
            return float(gp_predict(x, self.gp_model_single, self._x_scaler))

    def objective(self, trial):
        # Ask Optuna for a new x
        x1 = trial.suggest_float("x1", self.x_range[0], self.x_range[1])
        x2 = trial.suggest_float("x2", self.x_range[0], self.x_range[1])
        x_new = np.array([x1, x2])

        # Observe real (non-stationary) sources at current time t, store, refit model(s)
        self._observe_and_store(x_new, t=self.current_t)

        X_np = np.array(self.X_data)
        if self.aggregation_mode == "multi":
            Y_list_np = [np.array(y) for y in self.Y_data_list]
            self.gp_models = fit_multi_gps(X_np, Y_list_np, random_seed=self.random_seed, x_scaler=self._x_scaler)
        else:
            Y_mean_np = np.array(self.Y_mean)
            self.gp_model_single = fit_single_gp(X_np, Y_mean_np, random_seed=self.random_seed, x_scaler=self._x_scaler)

        # Return model-predicted mean (Optuna minimizes this)
        y_pred = self._predict_model(x_new)

        # For metrics: evaluate F_true at current t and compute distances to the moving optimum
        # Rosenbrock minimum at (1,1) in unshifted coords -> at s(t) + (1,1) when shifted
        s_now = self.s_t(self.current_t)
        x_opt_now = s_now + np.array([1.0, 1.0])
        f_true_now = self.F_true(x_new, self.current_t)
        dist_now = float(np.linalg.norm(x_new - x_opt_now))

        # Incumbent distance: best x (over all tried points) w.r.t. current t's optimal location
        X_all = np.array(self.X_data)
        dists_all = np.linalg.norm(X_all - x_opt_now, axis=1)
        incumbent_dist = float(np.min(dists_all))

        self.trial_records.append({
            "t": int(self.current_t),
            "x": x_new,
            "y_pred": y_pred,
            "f_true": f_true_now,
            "x_opt_t": x_opt_now,
            "dist_to_opt": dist_now,
            "incumbent_dist": incumbent_dist,
        })

        # Advance time AFTER finishing this trial
        self.current_t += 1

        return y_pred

    def run_optimization(self):
        # Run n_trials sequentially (time advances each trial)
        self.study.optimize(self.objective, n_trials=self.n_trials, n_jobs=1, catch=(Exception,))

        # Final best according to the study's model objective
        best_params = self.study.best_params
        best_value = self.study.best_value

        # For reporting: last-time-step optimum and the final evaluated point
        if len(self.trial_records) > 0:
            last_rec = self.trial_records[-1]
            f_true_found = last_rec["f_true"]
            x_min_local = last_rec["x_opt_t"]
        else:
            f_true_found = None
            x_min_local = None

        return best_params, best_value, None, f_true_found, x_min_local

    # Convenience: extract per-trial sequences for analysis
    def get_time_series(self):
        """Return dict of arrays: t, dist_to_opt, incumbent_dist, f_true_at_x, y_pred."""
        if len(self.trial_records) == 0:
            return None
        t = np.array([r["t"] for r in self.trial_records])
        dist_to_opt = np.array([r["dist_to_opt"] for r in self.trial_records])
        incumbent_dist = np.array([r["incumbent_dist"] for r in self.trial_records])
        f_true = np.array([r["f_true"] for r in self.trial_records])
        y_pred = np.array([r["y_pred"] for r in self.trial_records])
        return {
            "t": t,
            "dist_to_opt": dist_to_opt,
            "incumbent_dist": incumbent_dist,
            "f_true": f_true,
            "y_pred": y_pred,
        }

