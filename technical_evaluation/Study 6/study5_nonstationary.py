import numpy as np
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils_ns import MSourceBayesOpt

from tqdm import tqdm

if __name__ == "__main__":
    base_seed = 123

    # Experiment configuration
    x_range = [-2, 2]
    n_init = 10
    n_trials = 50

    # Non-stationary schedule: 5 blocks of 10 trials each
    shift_step_size = 10
    shift_per_step = (0.5, 0.5)
    start_shift = (-1.0, -1.0)

    # Constant number of sources for this study
    M = 3
    n_runs = 100

    # Precompute block slices (0-based indexing over the 50 trials)
    block_edges = [(i * shift_step_size, (i + 1) * shift_step_size) for i in range(n_trials // shift_step_size)]
    n_blocks = len(block_edges)  # should be 5

    # Storage: each is a list of length n_blocks; each element collects n_runs values
    single_mean_per_block = [ [] for _ in range(n_blocks) ]
    single_best_per_block = [ [] for _ in range(n_blocks) ]
    multi_mean_per_block  = [ [] for _ in range(n_blocks) ]
    multi_best_per_block  = [ [] for _ in range(n_blocks) ]

    print(f"Running {n_runs} runs (M={M}, n_init={n_init}, n_trials={n_trials}) ...")
    for run_idx in tqdm(range(n_runs), desc="Runs", unit="run"):
        seed = base_seed + 1000 * run_idx

        # Use the SAME source biases for both single and multi in this run
        rng_bias = np.random.default_rng(seed + 999)
        shared_biases = rng_bias.uniform(low=-1.0, high=1.0, size=(M, 2))

        # ---- SINGLE ----
        bo_single = MSourceBayesOpt(
            M=M,
            random_seed=seed,
            x_range=x_range,
            n_init=n_init,
            n_trials=n_trials,
            shift_step_size=shift_step_size,
            shift_per_step=shift_per_step,
            start_shift=start_shift,
            source_biases=shared_biases,
            aggregation_mode="single",
        )
        bo_single.run_optimization()
        ts_single = bo_single.get_time_series()  # keys: 't', 'dist_to_opt', 'incumbent_dist', ...

        # ---- MULTI ----
        bo_multi = MSourceBayesOpt(
            M=M,
            random_seed=seed,
            x_range=x_range,
            n_init=n_init,
            n_trials=n_trials,
            shift_step_size=shift_step_size,
            shift_per_step=shift_per_step,
            start_shift=start_shift,
            source_biases=shared_biases,
            aggregation_mode="multi",
        )
        bo_multi.run_optimization()
        ts_multi = bo_multi.get_time_series()

        # Distances to moving optimum per trial
        dist_single = np.asarray(ts_single["dist_to_opt"])  # length 50
        dist_multi  = np.asarray(ts_multi["dist_to_opt"])   # length 50

        # Aggregate per block: mean and best (min) distance within the 10-trial block
        for b, (lo, hi) in enumerate(block_edges):
            block_s = dist_single[lo:hi]
            block_m = dist_multi[lo:hi]

            single_mean_per_block[b].append(float(np.mean(block_s)))
            single_best_per_block[b].append(float(np.min(block_s)))

            multi_mean_per_block[b].append(float(np.mean(block_m)))
            multi_best_per_block[b].append(float(np.min(block_m)))

    # -------------------------------------------------------------------------
    # Plot boxplots: top = mean distance per block; bottom = best distance per block
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    blocks_labels = [f"{lo+1}-{hi}" for (lo, hi) in block_edges]
    axs[1].set_xticks(range(1, n_blocks + 1))
    axs[1].set_xticklabels(blocks_labels)
    axs[1].set_xlabel("Optimization trials")
    
    # positions: pair the boxes (single vs multi) at each block index
    pos_single = [i + 1 - 0.18 for i in range(n_blocks)]
    pos_multi  = [i + 1 + 0.18 for i in range(n_blocks)]

    # --- Top: MEAN distance per block ---
    mean_data_single = [single_mean_per_block[i] for i in range(n_blocks)]
    mean_data_multi  = [multi_mean_per_block[i]  for i in range(n_blocks)]
    axs[0].boxplot(mean_data_single, positions=pos_single, widths=0.32,
                   patch_artist=True, boxprops=dict(facecolor="lightcoral"),
                   medianprops=dict(color="darkred"), showfliers=False)
    axs[0].boxplot(mean_data_multi,  positions=pos_multi,  widths=0.32,
                   patch_artist=True, boxprops=dict(facecolor="lightblue"),
                   medianprops=dict(color="navy"), showfliers=False)
    axs[0].set_ylabel("Mean distance to moving optimum\n(10 trials per block)")
    axs[0].set_title(f"Study 5 — Distance to Moving Optimum, M={M}, {n_runs} runs")
    axs[0].grid(True, axis="y", linestyle="--", alpha=0.4)

    # --- Bottom: BEST (min) distance per block ---
    best_data_single = [single_best_per_block[i] for i in range(n_blocks)]
    best_data_multi  = [multi_best_per_block[i]  for i in range(n_blocks)]
    axs[1].boxplot(best_data_single, positions=pos_single, widths=0.32,
                   patch_artist=True, boxprops=dict(facecolor="lightcoral"),
                   medianprops=dict(color="darkred"), showfliers=False)
    axs[1].boxplot(best_data_multi,  positions=pos_multi,  widths=0.32,
                   patch_artist=True, boxprops=dict(facecolor="lightblue"),
                   medianprops=dict(color="navy"), showfliers=False)
    axs[1].set_ylabel("Best (min) distance in block")
    axs[1].grid(True, axis="y", linestyle="--", alpha=0.4)

    # Shared x-axis formatting
    axs[1].set_xticks(range(1, n_blocks + 1))
    axs[1].set_xticklabels(blocks_labels)

    # Legend
    legend_elems = [
        mpatches.Patch(facecolor="lightcoral", edgecolor="darkred", label="Single-surrogate"),
        mpatches.Patch(facecolor="lightblue",  edgecolor="navy",     label="Multi-surrogate"),
    ]
    axs[0].legend(handles=legend_elems, loc="upper left")

    plt.tight_layout()
    plt.savefig("study5_boxplots_mean_and_best_distance.svg")
    plt.savefig("study5_boxplots_mean_and_best_distance.png", dpi=200)
    plt.close()

    # Save raw data if you want to inspect later
    results = dict(
        config=dict(
            x_range=x_range,
            n_init=n_init,
            n_trials=n_trials,
            shift_step_size=shift_step_size,
            shift_per_step=shift_per_step,
            start_shift=start_shift,
            M=M,
            n_runs=n_runs,
        ),
        single_mean_per_block=single_mean_per_block,
        single_best_per_block=single_best_per_block,
        multi_mean_per_block=multi_mean_per_block,
        multi_best_per_block=multi_best_per_block,
    )
    with open("study5_boxplots_nonstationary_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("Saved figures: 'study5_boxplots_mean_and_best_distance.(svg|png)'")
    print("Saved raw: 'study5_boxplots_nonstationary_results.pkl'")
