# optimization.py
import numpy as np
import pickle

from utils import (
    TwoSourceBayesOpt,
    generate_weighted_sources,
    corrupted_objective_function,
    approximate_true_minimum,
)

if __name__ == "__main__":
    # Five weight matrices to test, each is [ [W11, W12], [W21, W22] ]
    W_conditions = [
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        #np.array([[0.7, 0.3], [0.2, 0.8]]),
        #np.array([[0.5, 0.5], [0.3, 0.7]]),
        #np.array([[0.8, 0.2], [0.6, 0.4]])
    ]

    x_range = [-3, 3]

    results = {}

    results['x_range'] = x_range

    # Random translations for the two Gaussians G1, G2 (fixed for all conditions)
    translations = np.random.uniform(-1, 1, size=(2, 2))

    # ------------------------------------------------------------------
    # Since F_TRUE is 0.5*(G1 + G2) and is the same for all conditions,
    # we can pre-compute its grid for plotting and store it in results.
    # ------------------------------------------------------------------
    # Approximate the global minimum on a finer grid:
    def F_true_global(x):
        # 0.5 * G1(x) + 0.5 * G2(x)
        return 0.5*corrupted_objective_function(x, translations[0]) + \
               0.5*corrupted_objective_function(x, translations[1])
    
        # return corrupted_objective_function(x, [0,0])

    x_min_true, f_min_true = approximate_true_minimum(F_true_global, x_range, n_grid=300)

    # Pre-compute a grid for F_TRUE in [-8,8] x [-8,8]
    res_global = 200
    x_lin_global = np.linspace(x_range[0], x_range[1], res_global)
    y_lin_global = np.linspace(x_range[0], x_range[1], res_global)
    Xg, Yg = np.meshgrid(x_lin_global, y_lin_global)
    Zg = np.zeros_like(Xg)
    for rr in range(res_global):
        for cc in range(res_global):
            pt = np.array([Xg[rr, cc], Yg[rr, cc]])
            Zg[rr, cc] = F_true_global(pt)

    # Save these into 'results' so we can retrieve them later for plotting
    results["F_true_grid"] = {
        "x_lin": x_lin_global,
        "y_lin": y_lin_global,
        "Z": Zg,
        "x_min_true": x_min_true,
        "f_min_true": f_min_true
    }
    # ------------------------------------------------------------------

    for idx, W_mat in enumerate(W_conditions):
        best_diff = float("inf")
        run_data = None
        all_diffs = []
        all_found_x = []

        seed_i = np.random.randint(0, 99999)

        # Build the two-source BO with this W
        bo_cond = TwoSourceBayesOpt(
            W_mat = W_mat,
            random_seed = seed_i,
            x_range = x_range,
            translations = translations,
            n_init = 10,         # Number of initial random samples
            n_trials = 50        # How many optimization steps
        )

        # Run the optimization
        found_params, found_value, f_true_min_val, f_true_found, x_min_local = bo_cond.run_optimization()

        # How close we got to the true global minimum of F_true
        diff = abs(f_true_found - f_true_min_val)
        all_diffs.append(diff)

        found_x = [found_params['x1'], found_params['x2']]
        all_found_x.append(found_x)

        run_data = {
                    "X_data": bo_cond.X_data,
                    "found_params": found_params,
                    "found_value": found_value,
                    "Z_grid_local": None,
                    "x_min_true": x_min_local,
                    "f_true_min_val": f_true_min_val
                }

        # Create contour data for F(x) = 0.5*F1(x) + 0.5*F2(x) in [-8,8]^2
        res_local = 200
        x1_lin_local = np.linspace(x_range[0], x_range[1], res_local)
        x2_lin_local = np.linspace(x_range[0], x_range[1], res_local)
        X_grid_local, Y_grid_local = np.meshgrid(x1_lin_local, x2_lin_local)
        Z_grid_local = np.zeros_like(X_grid_local)

        gp_models = bo_cond.refit_final_gp_models()
        for rr in range(res_local):
            for cc in range(res_local):
                x_pt_local = np.array([X_grid_local[rr, cc], Y_grid_local[rr, cc]])
                Z_grid_local[rr, cc] = bo_cond.predict_objective(x_pt_local, gp_models)

        run_data["Z_grid_local"] = Z_grid_local
        best_found_x = [
            run_data["found_params"]['x1'],
            run_data["found_params"]['x2']
        ]

        # Store everything for this condition
        results[idx] = {
            "W_mat": W_mat,
            "run_data": run_data,
            "all_diffs": all_diffs,
            "all_found_x": all_found_x,
            "best_found_x": best_found_x
        }

        # Save partial results after each condition
        with open("optimization_results.pkl", "wb") as f:
            pickle.dump(results, f)

    print("\nOptimization complete! Results saved to 'optimization_results.pkl'.")
