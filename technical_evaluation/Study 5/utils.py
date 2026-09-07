# utils.py
import random
import logging
import numpy as np
import optuna

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.optimize import fmin_l_bfgs_b


# =========================
# Fixed MinMax scaler (domain-based)
# =========================
class FixedMinMaxScaler:
    def __init__(self, low, high):
        self.low  = np.array(low, dtype=float)
        self.high = np.array(high, dtype=float)
        self.rng  = np.maximum(self.high - self.low, 1e-12)
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.low) / self.rng


# =========================
# Base functions
# =========================
def gaussian_2d(x):
    x1, x2 = x[0], x[1]
    return 1.0 - np.exp(-0.5 * (x1**2 + x2**2))

def gaussian_2d_translated(x, translation):
    return gaussian_2d(x - translation)

def rosenbrock(x):
    x1, x2 = x[0], x[1]
    return (1 - x1)**2 + 100.0 * (x2 - x1**2)**2

def three_hump_camel(x):
    x1, x2 = x[0], x[1]
    return 2*(x1**2) - 1.05*(x1**4) + (x1**6)/6 + x1*x2 + x2**2

def corrupted_objective_function(x, translation):
    return rosenbrock(x - translation)

def approximate_true_minimum(F_func, x_range, n_grid=200):
    x_space = np.linspace(x_range[0], x_range[1], n_grid)
    best_val, best_x = float("inf"), None
    for x1 in x_space:
        for x2 in x_space:
            pt = np.array([x1, x2])
            val = F_func(pt)
            if val < best_val:
                best_val, best_x = val, pt
    return best_x, best_val

def custom_optimizer(obj_func, initial_theta, bounds):
    x_opt, f_opt, _info = fmin_l_bfgs_b(
        func=obj_func,
        x0=initial_theta,
        bounds=bounds,
        maxiter=10000
    )
    return x_opt, f_opt


# =========================
# Source generation
# =========================
def generate_weighted_sources(W_mat, translations):
    """
    W_mat: (M, N)
    translations: (N, 2)

    G_j(x) = corrupted_objective_function(x, translations[j])
    F_i(x) = sum_j W_mat[i,j] * G_j(x)
    F_true(x) = mean_j G_j(x)  (reference)
    """
    M, N = W_mat.shape

    def make_G_func(shift):
        return lambda x, s=shift: corrupted_objective_function(x, s)

    G_funcs = [make_G_func(translations[j]) for j in range(N)]

    F_funcs = []
    for i in range(M):
        def make_f(i=i):
            return lambda x, i=i: sum(W_mat[i, j] * G_funcs[j](x) for j in range(N))
        F_funcs.append(make_f())

    def F_true(x):
        return np.mean([gj(x) for gj in G_funcs])

    return F_funcs, F_true, G_funcs


# =========================
# GP fitting
# =========================
def _default_kernel():
    return ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)

def fit_gps_multi(X_data, Y_cols, X_scaler, kernel=None, random_seed=None):
    """
    Fit one GP per source using a FIXED, domain-based X_scaler.
    Returns: list of GP models
    """
    kernel = kernel or _default_kernel()
    Xs = X_scaler.transform(X_data)

    models = []
    for y in Y_cols:
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=20,
            optimizer=custom_optimizer,
            random_state=random_seed,
        )
        gp.fit(Xs, y.reshape(-1))
        models.append(gp)
    return models

def fit_gp_single(X_data, Y_cols, X_scaler, kernel=None, random_seed=None):
    """
    Fit a single GP to the aggregated target (mean over sources),
    using the SAME fixed, domain-based scaler.
    """
    kernel = kernel or _default_kernel()
    Xs = X_scaler.transform(X_data)
    y_mean = np.mean(np.stack(Y_cols, axis=0), axis=0)

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=20,
        optimizer=custom_optimizer,
        random_state=random_seed,
    )
    gp.fit(Xs, y_mean.reshape(-1))
    return gp

def predict_gp_mean(x, gp_tuple, X_scaler):
    gp = gp_tuple
    xs = X_scaler.transform(x.reshape(1, -1))
    mu, _ = gp.predict(xs, return_std=True)
    return float(mu.reshape(-1)[0])


# =========================
# Unified BO class
# =========================
class MSourceBayesOpt:
    """
    mode='multi'  -> one GP per source, objective = mean of per-source means
    mode='single' -> single GP on aggregated target (mean over sources)
    """
    def __init__(
        self,
        W_mat,
        translations,
        random_seed=42,
        x_range=[-2, 2],
        n_init=10,
        n_trials=50,
        mode="multi",        # "multi" or "single"
        kernel=None
    ):
        assert mode in {"multi", "single"}, "mode must be 'multi' or 'single'"
        self.mode = mode
        self.kernel = kernel

        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.n_init = n_init
        self.n_trials = n_trials
        self.x_range = x_range

        # IMPORTANT: fixed, domain-based scaler (note the corrected lows/highs)
        x_min, x_max = self.x_range[0], self.x_range[1]
        self._x_scaler = FixedMinMaxScaler(low=[x_min, x_min], high=[x_max, x_max])

        # Build sources and reference
        self.F_funcs, self.F_true, self.G_funcs = generate_weighted_sources(W_mat, translations)
        self.M = len(self.F_funcs)

        # Seed initial data
        self.X_data = []
        self.Y_data = [[] for _ in range(self.M)]
        for _ in range(n_init):
            x1 = np.random.uniform(self.x_range[0], self.x_range[1])
            x2 = np.random.uniform(self.x_range[0], self.x_range[1])
            x = np.array([x1, x2])
            self.X_data.append(x)
            for i in range(self.M):
                self.Y_data[i].append(self.F_funcs[i](x))

        self.X_data = np.array(self.X_data)
        for i in range(self.M):
            self.Y_data[i] = np.array(self.Y_data[i])

        # Reference true min
        self.x_min_true, self.f_true_min_val = approximate_true_minimum(
            self.F_true, self.x_range, n_grid=200
        )

        # Optuna setup
        logging.getLogger("optuna").setLevel(logging.WARNING)
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=self.random_seed,
                n_startup_trials=20,
                n_ei_candidates=20,
                multivariate=True
            ),
        )

        # Model holders
        self.gp_models_multi = None   # list[(gp, X_scaler)]
        self.gp_model_single = None   # (gp, X_scaler)

    # ---------- internal ----------
    def _fit_models(self):
        if self.mode == "multi":
            self.gp_models_multi = fit_gps_multi(
                self.X_data, self.Y_data, X_scaler=self._x_scaler,
                kernel=self.kernel, random_seed=self.random_seed
            )
            self.gp_model_single = None
        else:
            self.gp_model_single = fit_gp_single(
                self.X_data, self.Y_data, X_scaler=self._x_scaler,
                kernel=self.kernel, random_seed=self.random_seed
            )
            self.gp_models_multi = None

    def _predict_objective(self, x):
        if self.mode == "multi":
            mus = [predict_gp_mean(x, tup, self._x_scaler) for tup in self.gp_models_multi]
            return float(np.mean(mus))
        else:
            return predict_gp_mean(x, self.gp_model_single, self._x_scaler)

    # ---------- optuna objective ----------
    def objective(self, trial):
        x1 = trial.suggest_float("x1", self.x_range[0], self.x_range[1])
        x2 = trial.suggest_float("x2", self.x_range[0], self.x_range[1])
        x_new = np.array([x1, x2])

        # Evaluate all sources
        Fi_vals = [f(x_new) for f in self.F_funcs]

        # Append
        self.X_data = np.vstack([self.X_data, x_new])
        for i in range(self.M):
            self.Y_data[i] = np.append(self.Y_data[i], Fi_vals[i])

        # Refit and predict
        self._fit_models()
        return self._predict_objective(x_new)

    # ---------- public ----------
    def run_optimization(self):
        # Initial fit
        self._fit_models()

        # BO loop
        self.study.optimize(self.objective, n_trials=self.n_trials, n_jobs=1, catch=(Exception,))

        best_value = self.study.best_value
        best_params = self.study.best_params

        # Evaluate F_true at best found point
        x_best = np.array([best_params["x1"], best_params["x2"]])
        f_true_found = self.F_true(x_best)

        return best_params, best_value, self.f_true_min_val, f_true_found, self.x_min_true
