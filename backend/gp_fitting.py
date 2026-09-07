# gp_fitting.py

import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C


##########################################################
# First up: Fitting a Gaussian Process model to our
# simulation data
##########################################################
def fit_simulation_gp(trials_dataset: pd.DataFrame, param_bounds, y_min, y_max):
    """
    Fits a GP model to our simulation values (from the 'simulated_based_value' column).

    Returns a tuple with two dictionaries:
        - r_preds: {trial_number: prediction} - the mean predictions
        - r_stds: {trial_number: std_dev} - how uncertain we are about each prediction
    """
    if trials_dataset.empty:
        return {}, {}

    # Grab the input features (X) and target values (y) from our dataset
    param_names = sorted(list(trials_dataset.iloc[0]["parameters"].keys()))
    X_list = []
    y_list = []
    trial_numbers = []

    for _, row in trials_dataset.iterrows():
        trial_numbers.append(row["trial_number"])
        param_dict = row["parameters"]
        X_list.append([param_dict[p] for p in param_names])
        y_list.append(row["simulated_based_value"])

    X = np.array(X_list)
    y = np.array(y_list).reshape(-1, 1)

    # Fixed [0,1] scaling for X using provided bounds
    x_low = np.array([param_bounds[n][0] for n in param_names], dtype=float)
    x_high = np.array([param_bounds[n][1] for n in param_names], dtype=float)
    x_rng = np.maximum(x_high - x_low, 1e-12)
    Xs = (X - x_low) / x_rng
    X_scaled = np.clip(Xs, 0.0, 1.0)

    # Fixed min–max scaling for y using provided range
    y_scaled = np.clip((y - y_min) / (y_max - y_min), 0.0, 1.0)

    # Special case: if all targets are identical, GP becomes pointless
    if len(np.unique(y_scaled)) == 1:
        r_predictions = {tn: float(y_scaled[0]) for tn in trial_numbers}
        # Zero uncertainty when all values are the same
        r_stds = {tn: 0.0 for tn in trial_numbers}
        return r_predictions, r_stds

    # Set up our kernel - Matern 2.5 works well for most engineering problems
    # We also throw in a constant kernel for flexibility
    kernel = C(1.0, (1e-3, 1e3)) * Matern(nu=2.5)

    # Create and fit our GP - using L-BFGS-B optimizer which usually works well
    gp = GaussianProcessRegressor(
        kernel=kernel, optimizer='fmin_l_bfgs_b', alpha=1e-6, normalize_y=False)
    gp.fit(X_scaled, y_scaled)

    # Predict at each trial (still in [0,1] units)
    r_predictions, r_stds = {}, {}
    for tn, row in zip(trial_numbers, trials_dataset.itertuples(index=False)):
        x_new = np.array([[row.parameters[n]
                         for n in param_names]], dtype=float)
        x_new_s = np.clip((x_new - x_low) / x_rng, 0.0, 1.0)
        mu_s, std_s = gp.predict(x_new_s, return_std=True)
        r_predictions[tn] = float(mu_s[0])
        r_stds[tn] = float(std_s[0])

    return r_predictions, r_stds


def fit_wave_gp(trials_dataset: pd.DataFrame,
                param_bounds,
                wave_metric_source: dict):
    """
    Fits a GP on the wave metric using the same recipe as fit_simulation_gp:
    - X: parameters scaled to [0,1] via param_bounds
    - y: wave metric (assumed in [0,1], clipped to [0,1])

    Returns:
        w_preds: {trial_number: mean}
        w_stds:  {trial_number: std}
    """
    if trials_dataset.empty or not wave_metric_source:
        return {}, {}

    # Build X (for trials that have wave metric) + y
    param_names = sorted(list(trials_dataset.iloc[0]["parameters"].keys()))
    X_list, y_list, tnums = [], [], []

    for _, row in trials_dataset.iterrows():
        tn = row["trial_number"]
        if tn in wave_metric_source and wave_metric_source[tn] is not None:
            p = row["parameters"]
            X_list.append([p[name] for name in param_names])
            # wave already ~[0,1]; clip to be safe
            y_list.append(float(np.clip(wave_metric_source[tn], 0.0, 1.0)))
            tnums.append(tn)

    if len(X_list) == 0:
        return {}, {}

    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=float).reshape(-1, 1)

    # Scale X to [0,1] with param bounds
    x_low = np.array([param_bounds[n][0] for n in param_names], dtype=float)
    x_high = np.array([param_bounds[n][1] for n in param_names], dtype=float)
    x_rng = np.maximum(x_high - x_low, 1e-12)
    Xs = np.clip((X - x_low) / x_rng, 0.0, 1.0)

    kernel = C(1.0, (1e-3, 1e3)) * Matern(nu=2.5)
    gp = GaussianProcessRegressor(
        kernel=kernel, optimizer='fmin_l_bfgs_b', alpha=1e-6, normalize_y=False)
    gp.fit(Xs, y)

    # Predict for *all* trials in trials_dataset (so fusion loop can just index)
    w_preds, w_stds = {}, {}
    for _, row in trials_dataset.iterrows():
        tn = row["trial_number"]
        x_new = np.array([[row["parameters"][n]
                         for n in param_names]], dtype=float)
        x_new_s = np.clip((x_new - x_low) / x_rng, 0.0, 1.0)
        mu_s, std_s = gp.predict(x_new_s, return_std=True)
        w_preds[tn] = float(mu_s[0])
        w_stds[tn] = float(std_s[0])
    return w_preds, w_stds


##########################################################
# Fitting a GP based on human preferences using
# a Plackett-Luce model
##########################################################
def fit_preference_gp(trials_dataset: pd.DataFrame,
                      user_preferences_dataset: pd.DataFrame,
                      param_bounds,
                      ):
    """
    Fits a GP on preferences (Plackett–Luce two-group) and optionally
    returns a fixed-scale [0,1] version using user-specified (g_min, g_max).

    If g_min and g_max are both provided:
        g_scaled = clip((g - g_min) / (g_max - g_min), clip_eps, 1-clip_eps)
        var_scaled = var / (g_max - g_min)^2
    Otherwise:
        returns the raw latent g (zero-meaned) and the unscaled approx variances.
    """
    # If we have no preference data or no trials, just return default values
    if user_preferences_dataset.empty or trials_dataset.empty:
        trial_numbers = trials_dataset["trial_number"].unique()
        default_value = 1.0 / \
            len(trial_numbers) if len(trial_numbers) > 0 else 0.5
        return {tn: default_value for tn in trial_numbers}, {tn: 1.0 for tn in trial_numbers}

    # 1) Index map
    trial_numbers_sorted = sorted(trials_dataset["trial_number"].unique())
    tn2idx = {tn: i for i, tn in enumerate(trial_numbers_sorted)}

    # 2) Feature matrix
    param_names = sorted(list(trials_dataset.iloc[0]["parameters"].keys()))
    X = np.array([[trials_dataset[trials_dataset["trial_number"] == tn].iloc[0]["parameters"][p]
                   for p in param_names] for tn in trial_numbers_sorted])

    # 3) Scale X to [0,1] using param_bounds (unchanged)
    x_low = np.array([param_bounds[n][0] for n in param_names], dtype=float)
    x_high = np.array([param_bounds[n][1] for n in param_names], dtype=float)
    x_rng = np.maximum(x_high - x_low, 1e-12)
    X_scaled = np.clip((X - x_low) / x_rng, 0.0, 1.0)

    # 4) Kernel matrix
    def matern52(xA, xB, length_scale=1.0, signal=1.0):
        r = np.sqrt(np.sum((xA - xB) ** 2)) / length_scale
        return signal * (1.0 + np.sqrt(5)*r + 5.0*r*r/3.0) * np.exp(-np.sqrt(5)*r)

    length_scale = 1.0
    signal = 1.0
    N = len(X_scaled)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = matern52(X_scaled[i], X_scaled[j], length_scale, signal)

    # 5) Build preference dataset
    D = []
    for _, row in user_preferences_dataset.iterrows():
        top_idxs = [tn2idx[tn] for tn in row["chosen_trials"]]
        bottom_idxs = [tn2idx[tn] for tn in row["not_chosen_trials"]]
        if len(top_idxs) == 0 and len(bottom_idxs) == 0:
            continue
        D.append([top_idxs, bottom_idxs])

    # 6) K^{-1}
    from scipy.linalg import cho_factor, cho_solve
    K_reg = K + 1e-7 * np.eye(N)
    cho = cho_factor(K_reg)
    K_inv = cho_solve(cho, np.eye(N))

    # 7) Two-group PL log-likelihood
    def pl_two_groups_loglik(g, top_idxs, bottom_idxs):
        numerator = np.sum(np.exp(g[top_idxs])) if len(top_idxs) > 0 else 0.0
        denominator = numerator + \
            (np.sum(np.exp(g[bottom_idxs])) if len(bottom_idxs) > 0 else 0.0)
        return np.log(numerator + 1e-12) - np.log(denominator + 1e-12)

    # 8) Posterior
    def log_posterior(g):
        ll = 0.0
        for top_idxs, bottom_idxs in D:
            ll += pl_two_groups_loglik(g, top_idxs, bottom_idxs)
        prior = -0.5 * g.dot(K_inv).dot(g)
        return ll + prior

    def neg_log_posterior(g):
        return -log_posterior(g)

    # 9) MAP
    from scipy.optimize import minimize
    g_init = np.zeros(N)
    res = minimize(neg_log_posterior, g_init, method='L-BFGS-B')
    g_map = res.x

    # 10) Package predictions and rough variances
    g_predictions_raw = {trial_numbers_sorted[i]: float(
        g_map[i]) for i in range(N)}

    preference_count = len(user_preferences_dataset)
    scaling_factor = 1.0 / (1.0 + preference_count)
    prior_variances = np.diag(K)
    posterior_variances = prior_variances * scaling_factor
    g_variances_raw = {trial_numbers_sorted[i]: float(
        posterior_variances[i]) for i in range(N)}

    return g_predictions_raw, g_variances_raw


##########################################################
# Fusion (now passes g_min/g_max through)
##########################################################
def fit_all_gp(trials_dataset: pd.DataFrame,
               user_preferences_dataset: pd.DataFrame,
               lambda_: float = 0.5,
               param_bounds=None,
               y_min=None,
               y_max=None,
               g_min: float = None,
               g_max: float = None,
               g_clip_eps: float = 0.0,
               wave_metric_source: dict | None = None,
               use_sources: list[str] | None = None,
               ):
    """
    - r(x): already mapped to [0,1] via (y_min,y_max) inside fit_simulation_gp
    - g(x): RAW latent from fit_preference_gp, then scaled HERE using (g_min,g_max)
    """
    if trials_dataset.empty:
        return {}, {}, {}, {}, {}

    if use_sources is None:
        use_sources = ["llm", "human"]  # legacy default

    use_llm = "llm" in use_sources
    use_human = "human" in use_sources
    use_wave = "wave" in use_sources and (wave_metric_source is not None)

    # Human-only selection?
    only_human = use_human and not use_llm and not use_wave

    # get models - always calculate all, but only use what's selected
    r_preds, r_stds = fit_simulation_gp(
        trials_dataset, param_bounds, y_min, y_max)

    g_raw,  g_var = fit_preference_gp(
        trials_dataset, user_preferences_dataset, param_bounds)

    w_preds, w_stds = ({}, {})
    if use_wave:
        w_preds, w_stds = fit_wave_gp(
            trials_dataset, param_bounds, wave_metric_source)

    # --- fixed external scaling for g -> [0,1] ---
    do_scale_human = (
                    (not only_human) and (not np.isclose(lambda_, 0.0)) and
                    (g_min is not None) and (g_max is not None) and (g_max > g_min)
    )

    def scale_g(mu, var, g_min, g_max, clip_eps):
        span = float(g_max - g_min)
        mu_s = (float(mu) - g_min) / span
        if clip_eps > 0.0:
            mu_s = float(np.clip(mu_s, clip_eps, 1.0 - clip_eps))
        var_s = float(var) / (span * span)
        return mu_s, var_s

    g_scaled, g_var_scaled = {}, {}
    any_scaled = False

    if do_scale_human:
        for tn, mu in g_raw.items():
            mu_s, var_s = scale_g(mu, g_var.get(tn, 1.0),
                                  g_min, g_max, g_clip_eps)
            g_scaled[tn] = mu_s
            g_var_scaled[tn] = var_s
        any_scaled = True
    else:
        # keep raw latent g (no scaling)
        g_scaled = dict(g_raw)
        g_var_scaled = dict(g_var)
        any_scaled = False

    # fuse on common scale if scaled; else lambda_ should be 0
    f_preds, disagreement, uncertainty = {}, {}, {}
    default_pref = 0.5 if any_scaled else 0.0

    for tn in r_preds.keys():
        r_val = float(r_preds[tn])                   # [0,1]
        # [0,1] if scaled else raw default
        g_val = float(g_scaled.get(tn, default_pref))
        w_val = None

        if use_wave and (tn in wave_metric_source) and (wave_metric_source[tn] is not None):
            w_val = float(w_preds[tn])

        # Legacy path when exactly {llm,human} - Also works for the tasks that are not Toggles!
        if use_llm and use_human and not use_wave:
            f = lambda_ * r_val + (1.0 - lambda_) * g_val
        else:
            chosen = []
            if use_llm:
                chosen.append(r_val)
            if use_human:
                chosen.append(g_val)
            if w_val is not None:
                chosen.append(w_val)
            f = float(np.mean(chosen)) if chosen else r_val

        f_preds[tn] = f

        # disagreement (only implemented for two sources)
        disagreement[tn] = abs(r_val - g_val) if any_scaled else 0.0
        r_var = (r_stds.get(tn, 1.0) ** 2)
        gv = g_var_scaled.get(tn, g_var.get(tn, 1.0))
        vars_ = [r_var, gv]
        if use_wave and (tn in w_stds):
            vars_.append(w_stds[tn] ** 2)
        uncertainty[tn] = float(np.mean(vars_))

    # return r (unit), g (scaled if requested), f, w (wave), disagreement, uncertainty
    return r_preds, (g_scaled if any_scaled else g_raw), f_preds, w_preds, disagreement, uncertainty
