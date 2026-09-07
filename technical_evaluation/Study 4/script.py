# ============================================
# Study 4: Optuna/TPE — Three Scenarios (near / medium / far)
#
# Output: Single PDF with THREE pages
#   • Page 1: Distance-to-current-optimum traces (all four methods)
#   • Page 2: Multi–Surrogate only — effect of Top-K seeding
#   • Page 3: Fused objectives before vs after reweighting
#
# Methods:
#   (A) Single — RESTART (discard past fused data)
#   (B) Single — CONTINUE (retain past fused labels under old weights)
#   (C) Single — SEEDS (restart; inject virtual reweighted past trials)
#   (D) Multi — SEEDS + fused-GP EI Top-K (restart; virtual, not budgeted)
#
# Metric: |x_best_so_far - x*_t|, where x*_t is the argmin of the
#         noise-free fused mean under the CURRENT weights at time t.
#         Best-so-far resets exactly at the reweighting time.
# ============================================

import warnings
import csv
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages

import optuna
from optuna.samplers import TPESampler
from optuna.trial import create_trial
from optuna.distributions import FloatDistribution

# Disable Optuna's logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Gaussian Processes
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

# EI utilities (normal pdf/cdf)
from scipy.stats import norm


# =============================
# Global configuration
# =============================
x_min, x_max = 0.0, 10.0
X_GRID = np.linspace(x_min, x_max, 2000).reshape(-1, 1)

# Timeline
T         = 40
CHANGE_AT = 21
T_PRE     = CHANGE_AT - 1
T_POST    = T - T_PRE

# BO / Monte Carlo
N_RUNS       = 100
BASE_SEED    = 1
N_STARTUP    = 5

# Multi-surrogate seeding
K_LEFT_MULTI  = 20               # K used for the main multi baseline (Page 1)
K_SWEEP_RIGHT = [20, 50, 100, 200, 300]  # K sweep (Page 2)

# Colors (consistent palette)
COLORS = {
    "multi_main":       "#0011FF",  # Multi
    "single_restart":   "#269826",  # Single (discard)
    "single_norestart": "#6B6B6B",  # Single (retain)
    "single_seeds":     "#D62728",  # Single (virtual reweighted)
}

# =============================
# Plot styling helper
# =============================
def style_axes_for_paper(ax):
    """Apply consistent, publication-style axis formatting."""
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.8)
        ax.spines[spine].set_color("#222")
    for spine in ["right", "top"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", which="both",
                   direction="out", length=6, width=1.6, color="#222",
                   labelsize=11, top=False, right=False)
    ax.grid(False)

# =============================
# Scenarios (fixed source functions; weights change only)
# =============================
def make_scenarios():
    """
    Define three weight-shift scenarios using the SAME source functions
    per scenario; only the weights (W_PRE, W_POST) differ.
    Each scenario controls how far the fused optimum moves at reweighting.
    """
    scenarios = []

    # Near shift
    def F_sim_mean_near(x):
        x = np.asarray(x, float); u = x
        out = 0.03 * (u - 5.0)**2
        out += -1.10 * np.exp(-0.5 * ((u - 5.0) / 0.50)**2)
        return out
    def F_des_mean_near(x):
        x = np.asarray(x, float); u = x
        out = 0.03 * (u - 5.0)**2
        out += -1.00 * np.exp(-0.5 * ((u - 5.6) / 0.55)**2)
        return out
    scenarios.append(dict(
        name="Near",
        F_sim_mean=F_sim_mean_near,
        F_des_mean=F_des_mean_near,
        noise_sim=0.08, noise_des=0.05,
        W_PRE=(0.55, 0.45), W_POST=(0.45, 0.55),
    ))

    # Medium shift
    def F_sim_mean_med(x):
        x = np.asarray(x, float); u = x
        out = 0.05 * (u - 5.0)**2
        out += -1.30 * np.exp(-0.5 * ((u - 3.2) / 0.40)**2)
        return out
    def F_des_mean_med(x):
        x = np.asarray(x, float); u = x
        out = 0.05 * (u - 5.0)**2
        out += -1.35 * np.exp(-0.5 * ((u - 7.1) / 0.45)**2)
        return out
    scenarios.append(dict(
        name="Medium",
        F_sim_mean=F_sim_mean_med,
        F_des_mean=F_des_mean_med,
        noise_sim=0.10, noise_des=0.05,
        W_PRE=(0.60, 0.40), W_POST=(0.35, 0.65),
    ))

    # Far shift
    def F_sim_mean_far(x):
        x = np.asarray(x, float); u = x
        out = 0.06 * (u - 5.0)**2
        out += -1.80 * np.exp(-0.5 * ((u - 1.5) / 0.28)**2)
        return out
    def F_des_mean_far(x):
        x = np.asarray(x, float); u = x
        out = 0.06 * (u - 5.0)**2
        out += -1.85 * np.exp(-0.5 * ((u - 8.6) / 0.30)**2)
        return out
    scenarios.append(dict(
        name="Far",
        F_sim_mean=F_sim_mean_far,
        F_des_mean=F_des_mean_far,
        noise_sim=0.12, noise_des=0.05,
        W_PRE=(0.80, 0.20), W_POST=(0.20, 0.80),
    ))

    return scenarios


# =============================
# Core math / BO helpers
# =============================
def new_gp(alpha):
    """Create a Matérn(ν=2.5) GP with output noise alpha, using scikit-learn."""
    k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5)
    return GaussianProcessRegressor(kernel=k, alpha=alpha,
                                    normalize_y=True, optimizer='fmin_l_bfgs_b')

def fused_true_mean(x, w, F_sim_mean, F_des_mean):
    """Noise-free fused mean under weights w=(w_s, w_d)."""
    return w[0]*F_sim_mean(x) + w[1]*F_des_mean(x)

def argmin_on_grid(arr, grid):
    """Return x at the minimum of a 1D array sampled on 'grid'."""
    idx = int(np.argmin(arr))
    return float(grid[idx, 0])

def distance_to_current_opt(x_best, w, F_sim_mean, F_des_mean):
    """|x_best_so_far - x*_t| for the CURRENT weights w."""
    f_grid = fused_true_mean(X_GRID, w, F_sim_mean, F_des_mean).ravel()
    x_star = argmin_on_grid(f_grid, X_GRID)
    return abs(float(x_best) - x_star)

def add_completed_trial(study, x, val):
    """Add a completed Optuna trial at x with objective value val."""
    trial = create_trial(
        params={"x": float(x)},
        distributions={"x": FloatDistribution(x_min, x_max)},
        value=float(val),
    )
    study.add_trial(trial)

def expected_improvement_min(mu, sigma, f_best, eps=1e-12):
    """
    EI for minimization using SciPy's normal CDF/PDF.
    EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z), z = (f_best - mu) / sigma.
    """
    sigma = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

def topk_from_grid(scores, grid, k, suppress_radius_frac=0.02):
    """
    Greedy Top-K argmax on 1D grid with local suppression for diversity.
    Used to pick EI peaks for virtual seeding.
    """
    scores = np.array(scores, float).copy()
    xs = grid.ravel()
    picks = []
    if k <= 0:
        return picks
    radius = suppress_radius_frac * (x_max - x_min)
    for _ in range(min(k, len(scores))):
        i = int(np.argmax(scores))
        if not np.isfinite(scores[i]) or scores[i] <= 0.0:
            break  # no more positive EI
        x_star = float(xs[i])
        picks.append((i, x_star))
        scores[np.abs(xs - x_star) <= radius] = -np.inf
    return picks


# =============================
# Scenario runner
# =============================
class ScenarioRunner:
    """
    Encapsulates a single scenario:
      - Fixed source means (F_sim_mean, F_des_mean) and noises
      - Weight schedule (W_PRE, W_POST)
      - Methods A/B/C/D as described at the top
    """

    def __init__(self, name, F_sim_mean, F_des_mean, noise_sim, noise_des, W_PRE, W_POST):
        self.name = name
        self.F_sim_mean = F_sim_mean
        self.F_des_mean = F_des_mean
        self.noise_sim = noise_sim
        self.noise_des = noise_des
        self.W_PRE = W_PRE
        self.W_POST = W_POST

    # --- Stochastic evaluations (with noise) ---
    def S_sim(self, x, rng):
        return self.F_sim_mean(x) + rng.normal(0, self.noise_sim, size=np.asarray(x).shape)

    def S_des(self, x, rng):
        return self.F_des_mean(x) + rng.normal(0, self.noise_des, size=np.asarray(x).shape)

    # --- Pre phase (shared) ---
    def run_pre_phase(self, seed):
        """Run TPE under W_PRE to collect pre-change samples for all methods."""
        rng = np.random.default_rng(seed)
        sampler = TPESampler(seed=seed, n_startup_trials=N_STARTUP)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        xs, ys_sim, ys_des, ys_fused = [], [], [], []
        W_PRE = self.W_PRE

        def objective(trial: optuna.trial.Trial):
            x = trial.suggest_float("x", x_min, x_max)
            y_s = self.S_sim(np.array([[x]]), rng).ravel()[0]
            y_d = self.S_des(np.array([[x]]), rng).ravel()[0]
            y_f = W_PRE[0]*y_s + W_PRE[1]*y_d
            xs.append(x); ys_sim.append(y_s); ys_des.append(y_d); ys_fused.append(y_f)
            return float(y_f)

        study.optimize(objective, n_trials=T_PRE, show_progress_bar=False)
        return np.array(xs).reshape(-1,1), np.array(ys_sim), np.array(ys_des), np.array(ys_fused)

    # --- Method A: Single — RESTART, no warm-start ---
    def restart_single_no_warmstart(self, seed):
        rng = np.random.default_rng(seed + 10_000)
        sampler = TPESampler(seed=seed, n_startup_trials=N_STARTUP)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        xs_post_real = []
        W_POST = self.W_POST

        def objective_post(trial: optuna.trial.Trial):
            x = trial.suggest_float("x", x_min, x_max)
            y = (W_POST[0]*self.S_sim(np.array([[x]]), rng) +
                 W_POST[1]*self.S_des(np.array([[x]]), rng)).ravel()[0]
            xs_post_real.append(x)
            return float(y)

        study.optimize(objective_post, n_trials=T_POST, show_progress_bar=False)
        return np.array(xs_post_real).reshape(-1,1)

    # --- Method B: Single — CONTINUE (retain fused labels under W_PRE) ---
    def continue_single_with_history(self, seed, xs_pre, y_fused_pre):
        rng = np.random.default_rng(seed + 11_000)
        sampler = TPESampler(seed=seed, n_startup_trials=N_STARTUP)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        for x, yf in zip(xs_pre.ravel(), y_fused_pre):
            add_completed_trial(study, x, float(yf))

        xs_post_real = []
        W_POST = self.W_POST
        def objective_post(trial: optuna.trial.Trial):
            x = trial.suggest_float("x", x_min, x_max)
            y = (W_POST[0]*self.S_sim(np.array([[x]]), rng) +
                 W_POST[1]*self.S_des(np.array([[x]]), rng)).ravel()[0]
            xs_post_real.append(x)
            return float(y)

        study.optimize(objective_post, n_trials=T_POST, show_progress_bar=False)
        return np.array(xs_post_real).reshape(-1,1)

    # --- Method C: Single — SEEDS (restart; virtual reweighted past) ---
    def restart_single_virtual_reweight_pre(self, seed, xs_pre, ys_sim_pre, ys_des_pre):
        rng = np.random.default_rng(seed + 12_000)
        sampler = TPESampler(seed=seed, n_startup_trials=N_STARTUP)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        # Virtual seeds at pre points using NEW weights
        y_virtual = self.W_POST[0]*ys_sim_pre + self.W_POST[1]*ys_des_pre
        for x, yv in zip(xs_pre.ravel(), y_virtual):
            add_completed_trial(study, x, float(yv))

        xs_post_real = []
        def objective_post(trial: optuna.trial.Trial):
            x = trial.suggest_float("x", x_min, x_max)
            y = (self.W_POST[0]*self.S_sim(np.array([[x]]), rng) +
                 self.W_POST[1]*self.S_des(np.array([[x]]), rng)).ravel()[0]
            xs_post_real.append(x)
            return float(y)

        study.optimize(objective_post, n_trials=T_POST, show_progress_bar=False)
        return np.array(xs_post_real).reshape(-1,1)

    # --- Method D: Multi — SEEDS + fused-GP EI Top-K (virtual; restart) ---
    def restart_multi_virtual_reweight_plus_topk(self, seed, xs_pre, ys_sim_pre, ys_des_pre, K_virtual):
        """
        Multi-surrogate warm-start at reweighting:
          (i)  Add reweighted past as *virtual* trials (NOT budgeted)
          (ii) Fit two independent GPs (sim/des) on pre data
          (iii) Build fused posterior on X_GRID under new weights
          (iv) Pick Top-K EI peaks directly from X_GRID (diversified)
          (v)  Add those as *virtual* trials with value = fused-mean mu_f
          (vi) Run T_POST real evaluations with the fused objective
        """
        rng = np.random.default_rng(seed + 20_000 + (K_virtual or 0))
        sampler = TPESampler(seed=seed, n_startup_trials=N_STARTUP)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        # (i) Virtual reweighted past
        y_virtual_past = self.W_POST[0]*ys_sim_pre + self.W_POST[1]*ys_des_pre
        for x, yv in zip(xs_pre.ravel(), y_virtual_past):
            add_completed_trial(study, x, float(yv))

        # (ii) Fit GPs
        gp_s = new_gp(alpha=max(self.noise_sim, 1e-6)**2).fit(xs_pre, ys_sim_pre)
        gp_d = new_gp(alpha=max(self.noise_des, 1e-6)**2).fit(xs_pre, ys_des_pre)

        # (iii) Fused posterior
        mu_s, std_s = gp_s.predict(X_GRID, return_std=True)
        mu_d, std_d = gp_d.predict(X_GRID, return_std=True)

        w_s, w_d = self.W_POST
        mu_f = w_s*mu_s + w_d*mu_d
        # assume independence between surrogates
        std_f = np.sqrt((w_s**2) * (std_s**2) + (w_d**2) * (std_d**2))

        # EI anchor: best (virtual) fused value at reweighting
        f_best = float(np.min(y_virtual_past)) if len(y_virtual_past) else np.inf
        ei = expected_improvement_min(mu_f, std_f, f_best)

        # (iv) Top-K EI peaks (with local suppression for diversity)
        picks = topk_from_grid(ei, X_GRID, k=(K_virtual or 0), suppress_radius_frac=0.02)

        # (v) Add fused-mean predictions at EI peaks as virtual seeds
        for idx, x_star in picks:
            add_completed_trial(study, float(x_star), float(mu_f[idx]))

        # (vi) Real post phase (counts towards budget)
        xs_post_real = []
        def objective_post(trial: optuna.trial.Trial):
            x = trial.suggest_float("x", x_min, x_max)
            y = (self.W_POST[0]*self.S_sim(np.array([[x]]), rng) +
                 self.W_POST[1]*self.S_des(np.array([[x]]), rng)).ravel()[0]
            xs_post_real.append(x)
            return float(y)

        study.optimize(objective_post, n_trials=T_POST, show_progress_bar=False)
        return np.array(xs_post_real).reshape(-1,1)

    # --- Distance trace utility (adds reweighting gap automatically) ---
    def dist_trace_from_xs(self, xs_pre, xs_post_real, rng_eval_seed):
        rng = np.random.default_rng(rng_eval_seed)
        y_pre  = (self.W_PRE[0]*self.S_sim(xs_pre, rng)        +
                  self.W_PRE[1]*self.S_des(xs_pre, rng)).ravel()
        y_post = (self.W_POST[0]*self.S_sim(xs_post_real, rng) +
                  self.W_POST[1]*self.S_des(xs_post_real, rng)).ravel()

        # Pre phase best-so-far under W_PRE
        best_val = np.inf; best_x = None
        dists_pre = []
        for x, y in zip(xs_pre.ravel(), y_pre):
            if y < best_val: best_val, best_x = y, float(x)
            dists_pre.append(distance_to_current_opt(best_x, self.W_PRE, self.F_sim_mean, self.F_des_mean))

        # Post phase best-so-far under W_POST (reset)
        best_val = np.inf; best_x = None
        dists_post = []
        for x, y in zip(xs_post_real.ravel(), y_post):
            if y < best_val: best_val, best_x = y, float(x)
            dists_post.append(distance_to_current_opt(best_x, self.W_POST, self.F_sim_mean, self.F_des_mean))

        out = np.array(dists_pre + dists_post)
        assert out.shape[0] == T
        return out

    # --- Fused objectives plot (noise-free means) ---
    def plot_fused_objectives(self, ax):
        x = X_GRID.ravel()
        F_pre  = fused_true_mean(X_GRID, self.W_PRE,  self.F_sim_mean, self.F_des_mean).ravel()
        F_post = fused_true_mean(X_GRID, self.W_POST, self.F_sim_mean, self.F_des_mean).ravel()
        i_pre  = int(np.argmin(F_pre));  i_post = int(np.argmin(F_post))
        x_pre_star, y_pre_star   = x[i_pre],  F_pre[i_pre]
        x_post_star, y_post_star = x[i_post], F_post[i_post]
        col_pre, col_post = COLORS["single_norestart"], COLORS["multi_main"]
        ax.plot(x, F_pre,  color=col_pre,  linewidth=2, label=f"Pre  w=({self.W_PRE[0]:.2f},{self.W_PRE[1]:.2f})")
        ax.plot(x, F_post, color=col_post, linewidth=2, label=f"Post w=({self.W_POST[0]:.2f},{self.W_POST[1]:.2f})")
        ax.plot([x_pre_star],  [y_pre_star],  marker="o", markersize=6, color=col_pre,  linestyle="None",
                label=fr"$x^\ast_{{\mathrm{{pre}}}}={x_pre_star:.2f}$")
        ax.plot([x_post_star], [y_post_star], marker="s", markersize=6, color=col_post, linestyle="None",
                label=fr"$x^\ast_{{\mathrm{{post}}}}={x_post_star:.2f}$")
        ax.set_xlim(x_min, x_max)
        ax.set_title(self.name)
        style_axes_for_paper(ax)


# =============================
# Run all scenarios & build plots (single PDF)
# =============================
def main():
    scenarios_cfg = make_scenarios()
    scenarios = [ScenarioRunner(**cfg) for cfg in scenarios_cfg]

    iters_pre  = np.arange(1, T_PRE+1)
    iters_post = np.arange(CHANGE_AT, T+1)

    # Storage: per scenario, each method -> (median, q1, q3)
    all_traces = []
    # Storage: per scenario, K -> (median, q1, q3) for Page 2
    all_multi_by_K = []

    with open("study_multi_scenarios_distances.csv", "w", newline="") as fd:
        writer = csv.DictWriter(fd, fieldnames=["scenario", "run", "method", "t", "dist"])
        writer.writeheader()

        for sc in scenarios:
            traces_A, traces_B, traces_C, traces_D = [], [], [], []
            traces_multi_by_K = {K: [] for K in K_SWEEP_RIGHT}

            for rid in [BASE_SEED + i for i in range(N_RUNS)]:
                xs_pre, ys_s_pre, ys_d_pre, y_f_pre = sc.run_pre_phase(rid)
                eval_seed = rid + 33_000  # shared evaluation noise across methods

                # A: single restart
                xs_post_A = sc.restart_single_no_warmstart(rid)
                d_A = sc.dist_trace_from_xs(xs_pre, xs_post_A, rng_eval_seed=eval_seed)
                traces_A.append(d_A)
                for t_idx, d in enumerate(d_A, 1):
                    writer.writerow({"scenario": sc.name, "run": rid, "method": "single_restart", "t": t_idx, "dist": float(d)})

                # B: single no-restart
                xs_post_B = sc.continue_single_with_history(rid, xs_pre, y_f_pre)
                d_B = sc.dist_trace_from_xs(xs_pre, xs_post_B, rng_eval_seed=eval_seed)
                traces_B.append(d_B)
                for t_idx, d in enumerate(d_B, 1):
                    writer.writerow({"scenario": sc.name, "run": rid, "method": "single_norestart", "t": t_idx, "dist": float(d)})

                # C: single seeds (virtual reweighted past)
                xs_post_C = sc.restart_single_virtual_reweight_pre(rid, xs_pre, ys_s_pre, ys_d_pre)
                d_C = sc.dist_trace_from_xs(xs_pre, xs_post_C, rng_eval_seed=eval_seed)
                traces_C.append(d_C)
                for t_idx, d in enumerate(d_C, 1):
                    writer.writerow({"scenario": sc.name, "run": rid, "method": "single_seeds_virtual", "t": t_idx, "dist": float(d)})

                # D: multi seeds + fused-GP EI Top-K (K_LEFT_MULTI)
                xs_post_D = sc.restart_multi_virtual_reweight_plus_topk(rid, xs_pre, ys_s_pre, ys_d_pre,
                                                                        K_virtual=K_LEFT_MULTI)
                d_D = sc.dist_trace_from_xs(xs_pre, xs_post_D, rng_eval_seed=eval_seed)
                traces_D.append(d_D)
                for t_idx, d in enumerate(d_D, 1):
                    writer.writerow({"scenario": sc.name, "run": rid, "method": "multi_topk_main", "t": t_idx, "dist": float(d)})

                # Page 2: sweep over K (multi only)
                for K in K_SWEEP_RIGHT:
                    xs_post_K = sc.restart_multi_virtual_reweight_plus_topk(rid, xs_pre, ys_s_pre, ys_d_pre,
                                                                            K_virtual=K)
                    d_K = sc.dist_trace_from_xs(xs_pre, xs_post_K, rng_eval_seed=eval_seed)
                    traces_multi_by_K[K].append(d_K)
                    for t_idx, d in enumerate(d_K, 1):
                        writer.writerow({"scenario": sc.name, "run": rid, "method": f"multi_k={K}", "t": t_idx, "dist": float(d)})

                print(f"[{sc.name}] run {rid} done")

            # Aggregate helper
            def med_iqr(A):
                A = np.asarray(A)
                return (np.median(A, axis=0),
                        np.percentile(A, 25, axis=0),
                        np.percentile(A, 75, axis=0))

            mA,q1A,q3A = med_iqr(np.vstack(traces_A))
            mB,q1B,q3B = med_iqr(np.vstack(traces_B))
            mC,q1C,q3C = med_iqr(np.vstack(traces_C))
            mD,q1D,q3D = med_iqr(np.vstack(traces_D))
            all_traces.append(dict(A=(mA,q1A,q3A), B=(mB,q1B,q3B),
                                   C=(mC,q1C,q3C), D=(mD,q1D,q3D)))

            agg_multi = {K: med_iqr(np.vstack(traces_multi_by_K[K])) for K in K_SWEEP_RIGHT}
            all_multi_by_K.append(agg_multi)

    # =============================
    # Create a single SVG (3 rows stacked)
    # =============================

    # --- Page 1: Distance traces (all methods) ---
    fig, axes = plt.subplots(3, 3, figsize=(16.0, 14.5), sharey='row')

    def plot_with_gap(ax, m, q1, q3, color, label, add_legend=False):
        ax.plot(iters_pre,  m[:T_PRE],         color=color, linewidth=2, label=label)
        ax.fill_between(iters_pre, q1[:T_PRE], q3[:T_PRE], color=color, alpha=0.18)
        m_post, q1_post, q3_post = m[CHANGE_AT-1:], q1[CHANGE_AT-1:], q3[CHANGE_AT-1:]
        ax.plot(iters_post, m_post,            color=color, linewidth=2, label="_nolegend_")
        ax.fill_between(iters_post, q1_post, q3_post,      color=color, alpha=0.18)
        if add_legend:
            ax.legend(loc="upper right", frameon=False, fontsize=9)

    # --- Row 1: Distance traces ---
    for i, sc in enumerate(scenarios):
        ax = axes[0, i]
        (mA,q1A,q3A) = all_traces[i]["A"]
        (mB,q1B,q3B) = all_traces[i]["B"]
        (mC,q1C,q3C) = all_traces[i]["C"]
        (mD,q1D,q3D) = all_traces[i]["D"]
        plot_with_gap(ax, mA,q1A,q3A, COLORS["single_restart"], "Single — No history")
        plot_with_gap(ax, mB,q1B,q3B, COLORS["single_norestart"], "Single — Old fused labels")
        plot_with_gap(ax, mC,q1C,q3C, COLORS["single_seeds"], "Single — Reweighted past labels")
        plot_with_gap(ax, mD,q1D,q3D, COLORS["multi_main"],
                    f"Multi — Reweighted past + EI seeds (K={K_LEFT_MULTI})",
                    add_legend=(i==2))
        ax.axvline(CHANGE_AT - 0.5, ls='--', c='k', alpha=0.7, label=("Reweighting" if i==2 else None))
        ax.set_title(sc.name)
        ax.set_xlabel("Optimization step (t)")
        if i == 0:
            ax.set_ylabel("Distance to current fused optimum")
        style_axes_for_paper(ax)

    axes[0,1].set_title("Distance to Optimum — Near / Medium / Far", fontsize=13)

    # --- Row 2: Multi–Surrogate only — Top-K sweep ---
    cmap = cm.get_cmap("Blues")
    K_sorted = sorted(K_SWEEP_RIGHT)
    def shade_for_rank(rank, total):
        a, b = 0.35, 0.85
        return cmap(a + (b - a) * (rank / max(1, total - 1)))

    for i, sc in enumerate(scenarios):
        ax = axes[1, i]
        colors_K = {K: COLORS["multi_main"] if K == K_LEFT_MULTI else shade_for_rank(idx, len(K_sorted))
                    for idx, K in enumerate(K_sorted)}
        for K in K_sorted:
            mK, q1K, q3K = all_multi_by_K[i][K]
            label = f"EI seeds (K={K})"
            ax.plot(iters_pre,  mK[:T_PRE],         color=colors_K[K], linewidth=2, label=label)
            ax.fill_between(iters_pre, q1K[:T_PRE], q3K[:T_PRE], color=colors_K[K], alpha=0.18)
            m_post, q1_post, q3_post = mK[CHANGE_AT-1:], q1K[CHANGE_AT-1:], q3K[CHANGE_AT-1:]
            ax.plot(iters_post, m_post,              color=colors_K[K], linewidth=2, label="_nolegend_")
            ax.fill_between(iters_post, q1_post, q3_post,        color=colors_K[K], alpha=0.18)
        ax.axvline(CHANGE_AT - 0.5, ls='--', c='k', alpha=0.7, label=("Reweighting" if i==2 else None))
        ax.set_title(f"{sc.name} — Effect of Top-K")
        ax.set_xlabel("Optimization step (t)")
        if i == 0:
            ax.set_ylabel("Distance to optimum")
        if i == 2:
            ax.legend(loc="upper right", frameon=False, fontsize=9)
        style_axes_for_paper(ax)

    # --- Row 3: Fused objectives ---
    for i, sc in enumerate(scenarios):
        ax = axes[2, i]
        sc.plot_fused_objectives(ax)
        ax.set_xlabel("Design variable $x$")
        if i == 0:
            ax.set_ylabel("Fused objective (noise-free mean)")
        ax.legend(loc="best", frameon=False, fontsize=9)
        style_axes_for_paper(ax)

    axes[2,1].set_title("Fused Objective — Before vs After", fontsize=13)

    plt.tight_layout()
    plt.savefig("study_scenarios_all_plots.svg", bbox_inches="tight", format="svg")
    plt.close(fig)
    print("Saved: study_scenarios_all_plots.svg")

    print("  - study_multi_scenarios_distances.csv")


if __name__ == "__main__":
    main()
