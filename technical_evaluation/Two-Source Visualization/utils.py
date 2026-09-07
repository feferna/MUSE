# utils.py
import numpy as np
import optuna
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.optimize import fmin_l_bfgs_b

# --------------------------------------------------------------------
# Base Gaussian in 2D, min = 0 at [0,0]
# --------------------------------------------------------------------
def gaussian_2d(x):
    x1, x2 = x[0], x[1]
    return 1.0 - np.exp(-0.5*(x1**2 + x2**2))

def gaussian_2d_translated(x, translation):
    return gaussian_2d(x - translation)

def three_hump_camel(x):
    x1, x2 = x[0], x[1]
    return (2*(x1**2)) - (1.05*(x1**4)) + ((x1**6)/6) + (x1*x2) + (x2**2)

def rosenbrock(x):
    x1, x2 = x[0], x[1]
    return (1 - x1)**2 + 100 * (x2 - x1**2)**2


def corrupted_objective_function(x, translation): 
    #return gaussian_2d_translated(x, translation)
    return rosenbrock(x - translation)

# --------------------------------------------------------------------
# Weighted source generation:
#   G1, G2 => F1(x) = W11*G1(x) + W12*G2(x)
#             F2(x) = W21*G1(x) + W22*G2(x)
# --------------------------------------------------------------------
def generate_weighted_sources(W_mat, translations, rng_seed):
    """
    W_mat is a 2x2 array: [[W11, W12],
                           [W21, W22]]
    translations is shape (2,2) => e.g. translations[0] for G1, translations[1] for G2
    Returns: 
      F1(x), F2(x), 
      plus a function F_true(x) = 0.5*(G1 + G2)
      and G1, G2 for reference.
    """
    rng = np.random.RandomState(rng_seed)

    def G1(x):
        return corrupted_objective_function(x, translations[0])
    def G2(x):
        return corrupted_objective_function(x, translations[1])

    # W11, W12, W21, W22
    W11, W12 = W_mat[0, 0], W_mat[0, 1]
    W21, W22 = W_mat[1, 0], W_mat[1, 1]

    def F1(x):
        return W11*G1(x) + W12*G2(x)

    def F2(x):
        return W21*G1(x) + W22*G2(x)

    def F_true(x):
        return 0.5*G1(x) + 0.5*G2(x)

    return F1, F2, F_true, G1, G2

# --------------------------------------------------------------------
# Approximate the minimum of F_true on a grid
# --------------------------------------------------------------------
def approximate_true_minimum(F_true, x_range, n_grid=200):
    x_space = np.linspace(x_range[0], x_range[1], n_grid)
    best_val = float("inf")
    best_x = None
    for x1 in x_space:
        for x2 in x_space:
            pt = np.array([x1, x2])
            val = F_true(pt)
            if val < best_val:
                best_val = val
                best_x = pt
    return best_x, best_val

# --------------------------------------------------------------------
# GP fitting functions
# --------------------------------------------------------------------
def custom_optimizer(obj_func, initial_theta, bounds):
    x_opt, f_opt, _info = fmin_l_bfgs_b(
        func=obj_func, x0=initial_theta, bounds=bounds, maxiter=10000
    )
    return x_opt, f_opt

def fit_gaussian_processes(X_data, Y_data, kernel=None):
    from sklearn.preprocessing import RobustScaler
    if kernel is None:
        kernel = ConstantKernel(1.0, (1e-4, 1e4)) * Matern(
            length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=2.5
        )
    X_scaler = RobustScaler()
    X_data_scaled = X_scaler.fit_transform(X_data)

    gp_models = []
    for y_vec in Y_data:
        Y_scaler = RobustScaler()
        y_scaled = Y_scaler.fit_transform(y_vec.reshape(-1, 1)).ravel()

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=False,
            n_restarts_optimizer=20,
            optimizer=custom_optimizer
        )
        gp.fit(X_data_scaled, y_scaled)
        gp_models.append((gp, X_scaler, Y_scaler))
    return gp_models

def predict_gp_value(x, gp_tuple):
    """
    gp_tuple = (gp, X_scaler, Y_scaler)
    """
    gp, X_scaler, Y_scaler = gp_tuple
    x_scaled = X_scaler.transform(x.reshape(1, -1))
    y_scaled_mean, _ = gp.predict(x_scaled, return_std=True)
    y_mean = Y_scaler.inverse_transform(y_scaled_mean.reshape(-1, 1)).ravel()[0]
    return y_mean

# --------------------------------------------------------------------
# Class for Two-Source Bayesian Optimization
# --------------------------------------------------------------------
class TwoSourceBayesOpt:
    def __init__(
        self,
        W_mat,
        translations,
        random_seed=42,
        x_range=[-20,20],
        n_init=10,
        n_trials=50
    ):
        """
        For each run:
          1) Generate G1, G2 from translations.
          2) F1, F2 using W_mat.
          3) F_true = 0.5*G1 + 0.5*G2
          4) Optimize F = 0.5*(F1 + F2) using a GP-based approach.
        """
        self.random_seed = random_seed
        self.n_init = n_init
        self.n_trials = n_trials
        self.x_range = x_range

        # Create F1, F2, F_true
        out = generate_weighted_sources(W_mat, translations, random_seed)
        self.F1, self.F2, self.F_true, self.G1, self.G2 = out

        # We'll store data for [F1, F2], i.e. two columns
        self.X_data = []
        self.Y_data = [[], []]  # Y_data[0] => F1, Y_data[1] => F2

        # Generate initial random points
        rng = np.random.RandomState(random_seed)
        for _ in range(n_init):
            x1 = rng.uniform(self.x_range[0], self.x_range[1])
            x2 = rng.uniform(self.x_range[0], self.x_range[1])
            x = np.array([x1, x2])
            self.X_data.append(x)
            self.Y_data[0].append(self.F1(x))  # F1
            self.Y_data[1].append(self.F2(x))  # F2

        self.X_data = np.array(self.X_data)
        self.Y_data[0] = np.array(self.Y_data[0])
        self.Y_data[1] = np.array(self.Y_data[1])

        # We also approximate min of F_true so we can measure error
        self.x_min_true, self.f_true_min_val = approximate_true_minimum(self.F_true, self.x_range, n_grid=300)

        # Prepare an optuna study to do sequential search
        self.study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=self.random_seed,
                n_startup_trials=20,
                n_ei_candidates=20,
                multivariate=True
            )
        )
        self.gp_models = None

    def objective(self, trial):
        # Propose a new point x
        x1 = trial.suggest_float("x1", self.x_range[0], self.x_range[1])
        x2 = trial.suggest_float("x2", self.x_range[0], self.x_range[1])
        x_new = np.array([x1, x2])

        # Evaluate F1, F2 at x_new
        val1 = self.F1(x_new)
        val2 = self.F2(x_new)

        # Update data
        self.X_data = np.vstack([self.X_data, x_new])
        self.Y_data[0] = np.append(self.Y_data[0], val1)
        self.Y_data[1] = np.append(self.Y_data[1], val2)

        # Refit GPs for [F1, F2]
        self.gp_models = fit_gaussian_processes(self.X_data, self.Y_data)

        # Evaluate the objective: F(x) = 0.5 * F1(x) + 0.5 * F2(x)
        f_val = self.predict_objective(x_new, self.gp_models)
        return f_val

    def predict_objective(self, x, gp_models):
        mean_1 = predict_gp_value(x, gp_models[0])  # F1
        mean_2 = predict_gp_value(x, gp_models[1])  # F2
        return 0.5*mean_1 + 0.5*mean_2

    def predict_true(self, x, gp_models):
        """
        Evaluate F_true(x) = 0.5*G1 + 0.5*G2,
        but we can approximate it from the learned GPs for F1, F2 
        or just call the actual function self.F_true(x).
        For consistency, let's call the real function, not the GPs:
        """
        return self.F_true(x)

    def run_optimization(self):
        # Optimize for self.n_trials
        self.study.optimize(self.objective, n_trials=self.n_trials, n_jobs=1, catch=(Exception,))

        # Retrieve best
        best_value = self.study.best_value
        best_params = self.study.best_params
        print("\nOptimization finished!")
        print(f"  => Best (combined) objective: {best_value:.4f}")
        print(f"  => Best parameters: {best_params}")

        # Evaluate F_true at the found best x
        x_best = np.array([best_params["x1"], best_params["x2"]])
        gp_models = self.refit_final_gp_models()
        # Or simply call F_true directly
        f_true_found = self.predict_true(x_best, gp_models)

        return best_params, best_value, self.f_true_min_val, f_true_found, self.x_min_true

    def refit_final_gp_models(self):
        """
        Make sure GPs are fitted with the final data so we can do plotting.
        """
        if self.gp_models is None:
            self.gp_models = fit_gaussian_processes(self.X_data, self.Y_data)
        return self.gp_models
