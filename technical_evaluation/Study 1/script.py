import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

# Increase all font sizes in the plots
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13
})

# ---------------------------
# Base functions
# ---------------------------
def f1(x):
    return np.sin(x)

def f2(x):
    return np.cos(x)

def F_true(x):
    return 0.5 * f1(x) + 0.5 * f2(x)

# Noise-parameterized sources (std dev = sigma)
def F_sim(x, sigma):
    noise = np.random.normal(0, sigma, size=x.shape)
    return 0.7 * f1(x) + 0.0 * f2(x) + noise

def F_user(x, sigma):
    noise = np.random.normal(0, sigma, size=x.shape)
    return 0.4 * f1(x) + 0.6 * f2(x) + noise

def F_expert(x, sigma):
    noise = np.random.normal(0, sigma, size=x.shape)
    return 0.2 * f1(x) + 0.8 * f2(x) + noise

# ---------------------------
# One run of a scenario at a given seed
# ---------------------------
def run_scenario(name, sigma_sim, sigma_user, sigma_expert, make_plot=False, seed=42):
    x_min, x_max = 0, 10
    np.random.seed(seed)

    # Sampling schedules (unchanged)
    x_sim   = np.linspace(x_min, x_max, 100)  # Simulator (dense)
    x_user_ = np.linspace(x_min, x_max, 10)   # User (sparse)
    x_exp   = np.linspace(x_min, x_max, 20)   # Expert (intermediate)

    # Generate data with chosen noise
    y_sim   = F_sim(x_sim, sigma_sim)
    y_user  = F_user(x_user_, sigma_user)
    y_expert= F_expert(x_exp, sigma_expert)

    # Kernel (unchanged)
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)

    # Fit individual GPs with alpha = sigma^2
    gp_sim = GaussianProcessRegressor(kernel=kernel, alpha=sigma_sim**2, normalize_y=True)
    gp_sim.fit(x_sim.reshape(-1, 1), y_sim)

    gp_user = GaussianProcessRegressor(kernel=kernel, alpha=sigma_user**2, normalize_y=True)
    gp_user.fit(x_user_.reshape(-1, 1), y_user)

    gp_expert = GaussianProcessRegressor(kernel=kernel, alpha=sigma_expert**2, normalize_y=True)
    gp_expert.fit(x_exp.reshape(-1, 1), y_expert)

    # Prediction grid
    x_grid = np.linspace(x_min, x_max, 200).reshape(-1, 1)
    y_true = F_true(x_grid.ravel())

    # Multi-surrogate predictions (unchanged logic)
    y_pred_sim,  sigma_sim_pred  = gp_sim.predict(x_grid, return_std=True)
    y_pred_user, sigma_user_pred = gp_user.predict(x_grid, return_std=True)
    y_pred_exp,  sigma_exp_pred  = gp_expert.predict(x_grid, return_std=True)

    y_combined = (y_pred_sim + y_pred_user + y_pred_exp) / 3.0
    combined_sigma = (sigma_sim_pred + sigma_user_pred + sigma_exp_pred) / 3.0
    mse_multi = np.mean((y_combined - y_true) ** 2)

    # Single-surrogate baseline: evaluate all three at user points and average (unchanged)
    y_sim_at_user    = F_sim(x_user_, sigma_sim)
    y_user_at_user   = F_user(x_user_, sigma_user)
    y_expert_at_user = F_expert(x_user_, sigma_expert)
    y_combined_single = (y_sim_at_user + y_user_at_user + y_expert_at_user) / 3.0

    alpha_single = (sigma_sim**2 + sigma_user**2 + sigma_expert**2) / 3.0
    gp_single = GaussianProcessRegressor(kernel=kernel, alpha=alpha_single, normalize_y=True)
    gp_single.fit(x_user_.reshape(-1, 1), y_combined_single)
    y_pred_single, sigma_single = gp_single.predict(x_grid, return_std=True)
    mse_single = np.mean((y_pred_single - y_true) ** 2)

    # Optional figure
    if make_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharey=True)

        # --- Top: Single Surrogate ---
        ax1.plot(x_grid, y_true, 'k--', label="True Function")
        ax1.plot(x_grid, y_pred_single, color='darkorange', linewidth=2, label="Single Surrogate GP")
        ax1.fill_between(x_grid.ravel(),
                         y_pred_single - sigma_single,
                         y_pred_single + sigma_single,
                         color='darkorange', alpha=0.2, label="Uncertainty")
        ax1.scatter(x_user_, y_combined_single, c='purple', marker='o', s=60,
                    label="Combined Samples\n(at slowest points)")
        ax1.set_title(f"Single Surrogate (Fused at Slowest Points) — {name}")
        ax1.set_xlabel("x"); ax1.set_ylabel("f(x)"); ax1.legend()

        # --- Middle: Multi-Surrogate Details ---
        ax2.plot(x_grid, y_true, 'k--', label="True Function")
        ax2.plot(x_grid, y_pred_sim, 'g-', label="Simulator GP")
        ax2.plot(x_grid, y_pred_user, 'r-', label="User GP")
        ax2.plot(x_grid, y_pred_exp, 'm-', label="Expert GP")
        ax2.plot(x_grid, y_combined, 'b-', linewidth=2, label="Multi-Surrogate GP")
        ax2.fill_between(x_grid.ravel(),
                         y_combined - combined_sigma,
                         y_combined + combined_sigma,
                         color='blue', alpha=0.2, label="Combined Uncertainty")
        ax2.scatter(x_sim, y_sim, c='g', marker='x', label="Simulator Samples")
        ax2.scatter(x_user_, y_user, c='r', marker='o', label="User Samples")
        ax2.scatter(x_exp, y_expert, c='m', marker='s', label="Expert Samples")
        ax2.set_title("Multi-Surrogate GP: Individual Surrogates and Combined Prediction")
        ax2.set_xlabel("x"); ax2.set_ylabel("f(x)"); ax2.legend()

        # --- Bottom: Comparison ---
        ax3.plot(x_grid, y_true, 'k--', label="True Function")
        ax3.plot(x_grid, y_pred_single, color='darkorange', linewidth=2,
                 label=f"Single Surrogate GP (MSE = {mse_single:.4f})")
        ax3.fill_between(x_grid.ravel(),
                         y_pred_single - sigma_single,
                         y_pred_single + sigma_single,
                         color='darkorange', alpha=0.2)
        ax3.plot(x_grid, y_combined, 'b-', linewidth=2,
                 label=f"Multi-Surrogate GP (MSE = {mse_multi:.4f})")
        ax3.fill_between(x_grid.ravel(),
                         y_combined - combined_sigma,
                         y_combined + combined_sigma,
                         color='blue', alpha=0.2)
        ax3.set_title("Comparison: Single vs. Multi-Surrogate GP")
        ax3.set_xlabel("x"); ax3.set_ylabel("f(x)"); ax3.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(f"combined_three_plots_{name.replace(' ', '_').lower()}.pdf", dpi=300)
        plt.close()

    return {
        "scenario": name,
        "sigma_sim": sigma_sim,
        "sigma_user": sigma_user,
        "sigma_expert": sigma_expert,
        "mse_single": mse_single,
        "mse_multi": mse_multi,
        "gain_%": 100.0 * (mse_single - mse_multi) / (mse_single + 1e-12)
    }

# ---------------------------
# Run each scenario across many seeds and summarize
# ---------------------------
def run_scenario_over_seeds(name, sigma_sim, sigma_user, sigma_expert,
                            n_seeds=100, make_plot_first=False, base_seed=42):
    rows = []
    for i in range(n_seeds):
        seed = base_seed + i
        res = run_scenario(
            name, sigma_sim, sigma_user, sigma_expert,
            make_plot=(make_plot_first and i == 0),
            seed=seed
        )
        res["seed"] = seed
        rows.append(res)
    df = pd.DataFrame(rows)

    # Paired diff per seed
    d = df["mse_single"] - df["mse_multi"]
    gain = 100.0 * d / (df["mse_single"] + 1e-12)
    n = len(df)
    d_mean = d.mean()
    d_std = d.std(ddof=1)
    ci_hw = 1.96 * d_std / np.sqrt(n)

    stats = {
        "scenario": name,
        "sigma_sim": sigma_sim,
        "sigma_user": sigma_user,
        "sigma_expert": sigma_expert,
        "seeds": n,
        "single_mean": df["mse_single"].mean(),
        "single_std": df["mse_single"].std(ddof=1),
        "single_median": df["mse_single"].median(),
        "single_q1": df["mse_single"].quantile(0.25),
        "single_q3": df["mse_single"].quantile(0.75),
        "multi_mean": df["mse_multi"].mean(),
        "multi_std": df["mse_multi"].std(ddof=1),
        "multi_median": df["mse_multi"].median(),
        "multi_q1": df["mse_multi"].quantile(0.25),
        "multi_q3": df["mse_multi"].quantile(0.75),
        "delta_mean": d_mean,
        "delta_std": d_std,
        "delta_ci95_low": d_mean - ci_hw,
        "delta_ci95_high": d_mean + ci_hw,
        "gain_mean_%": gain.mean(),
        "gain_std_%": gain.std(ddof=1),
        "win_rate_%": 100.0 * (d > 0).mean(),  # % seeds where multi beats single
        "ratio_mean": (df["mse_multi"] / (df["mse_single"] + 1e-12)).mean()
    }
    return df, pd.DataFrame([stats])

# ---------------------------
# Sensitivity + robustness study
# ---------------------------
if __name__ == "__main__":
    N_SEEDS = 100  # <-- set how many seeds you want

    # Scenarios: (name, sigma_sim, sigma_user, sigma_expert)
    scenarios = [
        ("Baseline",        0.20, 0.05, 0.10),
        ("Noisy simulator", 0.40, 0.05, 0.10),
        ("Noisy user",      0.20, 0.15, 0.10),
        ("Noisy expert",    0.20, 0.05, 0.25),
        ("All low",         0.05, 0.02, 0.05),
        ("All high",        0.30, 0.20, 0.25),
    ]

    all_raw = []
    all_stats = []
    for i, (name, s_sim, s_user, s_exp) in enumerate(scenarios):
        df_raw, df_stats = run_scenario_over_seeds(
            name, s_sim, s_user, s_exp,
            n_seeds=N_SEEDS,
            make_plot_first=(i == 0),  # only plot the first scenario (seed 42)
            base_seed=42
        )
        all_raw.append(df_raw)
        all_stats.append(df_stats)

    raw = pd.concat(all_raw, ignore_index=True)
    summary = pd.concat(all_stats, ignore_index=True)

    # Sort summary by multi_mean (best first)
    summary = summary.sort_values(by="multi_mean").reset_index(drop=True)

    # Column order for readability
    cols = [
        "scenario", "seeds",
        "sigma_sim", "sigma_user", "sigma_expert",
        "single_mean", "single_std", "single_median", "single_q1", "single_q3",
        "multi_mean", "multi_std", "multi_median", "multi_q1", "multi_q3",
        "delta_mean", "delta_std", "delta_ci95_low", "delta_ci95_high",
        "gain_mean_%", "gain_std_%", "win_rate_%", "ratio_mean"
    ]
    summary = summary[cols]

    # Pretty print to console
    with pd.option_context('display.float_format', '{:,.5f}'.format):
        print("\n=== Robustness across seeds (N = {}) ===".format(N_SEEDS))
        print(summary.to_string(index=False))

    # Save artifacts
    raw.to_csv("sensitivity_results_raw.csv", index=False)
    summary.to_csv("sensitivity_results_summary.csv", index=False)
    print("\nSaved raw per-seed results to sensitivity_results_raw.csv")
    print("Saved scenario summary to sensitivity_results_summary.csv")
    print("Saved 3-panel figure for the first scenario (seed=42) as combined_three_plots_baseline.pdf")
