# run_bo.py
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from utils import (
    MSourceBayesOpt,
    corrupted_objective_function,
    approximate_true_minimum,
)

def run_experiments(
    mode: str,
    base_seed: int,
    M_sources_list,
    N_objectives: int,
    x_range,
    n_runs_per_M: int,
    n_init: int,
    n_trials: int,
    n_grid_true_min: int,
    out_path: str,
):
    assert mode in {"multi", "single"}, "mode must be 'multi' or 'single'"

    results = {"x_range": x_range, "mode": mode}

    # Fixed translations for all runs
    rng_translations = np.random.default_rng(base_seed)
    translations = rng_translations.uniform(x_range[0], x_range[1], size=(N_objectives, 2))
    results["translations"] = translations

    # Define F_true(x) = mean_j G_j(x) for reference and approximate its min once
    def F_true_global(x):
        vals = [corrupted_objective_function(x, translations[j]) for j in range(N_objectives)]
        return np.mean(vals)

    x_min_true, f_min_true = approximate_true_minimum(F_true_global, x_range, n_grid=n_grid_true_min)
    results["x_min_true"] = x_min_true
    results["f_min_true"] = f_min_true

    distance_dict = {}

    for M in tqdm(M_sources_list, desc=f"[{mode}] M values"):
        dist_list = []

        for run_idx in tqdm(range(n_runs_per_M), desc=f"Runs for M={M}", leave=False):
            current_seed = base_seed + run_idx

            # New weights per run
            rng = np.random.default_rng(current_seed)
            W_mat = rng.uniform(low=0.0, high=1.0, size=(M, N_objectives))

            # Build optimizer in selected mode
            bo = MSourceBayesOpt(
                W_mat=W_mat,
                translations=translations,
                random_seed=current_seed,
                x_range=x_range,
                n_init=n_init,
                n_trials=n_trials,
                mode=mode,               # "multi" or "single"
            )

            best_params, best_value, f_true_min_val, f_true_found, x_min_local = bo.run_optimization()

            # Distance to the precomputed global min
            x_best = np.array([best_params["x1"], best_params["x2"]])
            dist = np.linalg.norm(x_best - x_min_true)
            dist_list.append(dist)

        distance_dict[f"M={M}"] = dist_list
        results["distances"] = distance_dict

        with open(out_path, "wb") as f:
            pickle.dump(results, f)

    print(f"\n[{mode}] All optimizations complete. Results saved to '{out_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi vs single surrogate BO.")
    parser.add_argument("--mode", choices=["multi", "single", "both"], required=True)
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--M-list", type=int, nargs="+", default=[1, 2, 5, 8, 10])
    parser.add_argument("--N-objectives", type=int, default=15)
    parser.add_argument("--x-min", type=float, default=-2.0)
    parser.add_argument("--x-max", type=float, default=2.0)
    parser.add_argument("--runs-per-M", type=int, default=100)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-grid-true-min", type=int, default=300)
    parser.add_argument("--out", type=str, default="optimization_results.pkl",
                        help="If mode=both, this becomes a prefix like results -> results_multi.pkl and results_single.pkl")
    args = parser.parse_args()

    x_range = [args.x_min, args.x_max]

    if args.mode in ("multi", "single"):
        out_path = args.out
        if out_path.endswith(".pkl"):
            out_path = out_path
        else:
            out_path = out_path + ".pkl"

        run_experiments(
            mode=args.mode,
            base_seed=args.base_seed,
            M_sources_list=args.M_list,
            N_objectives=args.N_objectives,
            x_range=x_range,
            n_runs_per_M=args.runs_per_M,
            n_init=args.n_init,
            n_trials=args.n_trials,
            n_grid_true_min=args.n_grid_true_min,
            out_path=out_path,
        )

    else:
        # both
        base = args.out[:-4] if args.out.endswith(".pkl") else args.out
        out_multi = f"{base}_multi.pkl"
        out_single = f"{base}_single.pkl"

        run_experiments(
            mode="multi",
            base_seed=args.base_seed,
            M_sources_list=args.M_list,
            N_objectives=args.N_objectives,
            x_range=x_range,
            n_runs_per_M=args.runs_per_M,
            n_init=args.n_init,
            n_trials=args.n_trials,
            n_grid_true_min=args.n_grid_true_min,
            out_path=out_multi,
        )
        run_experiments(
            mode="single",
            base_seed=args.base_seed,
            M_sources_list=args.M_list,
            N_objectives=args.N_objectives,
            x_range=x_range,
            n_runs_per_M=args.runs_per_M,
            n_init=args.n_init,
            n_trials=args.n_trials,
            n_grid_true_min=args.n_grid_true_min,
            out_path=out_single,
        )
