import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Increase all font sizes in the plots
plt.rcParams.update({
    'font.size': 14,       # Base font size
    'axes.labelsize': 16,  # Axis label size
    'axes.titlesize': 16,  # Title font size
    'legend.fontsize': 14, # Legend font size
    'xtick.labelsize': 13,
    'ytick.labelsize': 13
})

with open("optimization_results.pkl", "rb") as f:
    results = pickle.load(f)

condition_indices = [k for k in results.keys() if isinstance(k, int)]
sorted_indices = sorted(condition_indices)
x_range = results['x_range']

for idx in sorted_indices:
    cond_data = results[idx]
    W_mat = cond_data["W_mat"]
    run_data = cond_data["run_data"]
    X_data = run_data["X_data"]
    best_params = run_data["found_params"]
    Z_grid_local = run_data["Z_grid_local"]
    x_min_true = run_data["x_min_true"]
    best_x_found = np.array([best_params["x1"], best_params["x2"]])
    
    if Z_grid_local is None:
        continue

    # Find the min; if <= 0, shift everything
    raw_min = Z_grid_local.min()
    if raw_min <= 0:
        shift = abs(raw_min) + 1e-8  # just enough so min is now > 0
    else:
        shift = 1e-8  # small shift to avoid log(0) if there's an exact zero

    Z_grid_log = Z_grid_local + shift  # shift entire grid

    res_local = Z_grid_local.shape[0]
    x1_lin_local = np.linspace(x_range[0], x_range[1], res_local)
    x2_lin_local = np.linspace(x_range[0], x_range[1], res_local)
    X_grid_local, Y_grid_local = np.meshgrid(x1_lin_local, x2_lin_local)

    plt.figure(figsize=(6, 5))

    cs_local = plt.contourf(
        X_grid_local, Y_grid_local, Z_grid_log,
        levels=50,
        cmap="viridis",
        norm=LogNorm(
            vmin=Z_grid_log.min(),
            vmax=Z_grid_log.max()
        )
    )
    cs_lines = plt.contour(
        X_grid_local, Y_grid_local, Z_grid_log,
        levels=50,
        colors='white',
        linewidths=0.3,
        norm=LogNorm(
            vmin=Z_grid_log.min(),
            vmax=Z_grid_log.max()
        )
    )
    plt.colorbar(cs_local, label=r"$F(x)$ (log scale)")

    plt.scatter(X_data[:, 0], X_data[:, 1],
                c="white", s=20, edgecolors="black",
                label="Sampled Points")
    plt.scatter(best_x_found[0], best_x_found[1],
                c="red", marker="x", s=120,
                label="Found Best")
    plt.scatter(x_min_true[0], x_min_true[1],
                c="blue", marker="o", s=70,
                label=r"True Min of $F_{\mathrm{true}}$")

    plt.title(f"Condition {idx+1}")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"contour_best_run_condition_{idx+1}.pdf", dpi=300)
    plt.close()

print("Log-scale contour plots have been generated!")

# --------------------------------------------------------
# 3) For F_true (same fix)
# --------------------------------------------------------
F_true_grid = results["F_true_grid"]
x_lin = F_true_grid["x_lin"]
y_lin = F_true_grid["y_lin"]
Z = F_true_grid["Z"]
x_min_true = F_true_grid["x_min_true"]
f_min_true = F_true_grid["f_min_true"]

raw_min = Z.min()
if raw_min <= 0:
    shift = abs(raw_min) + 1e-8
else:
    shift = 1e-8

Z_log = Z + shift

plt.figure(figsize=(7, 6))
cs = plt.contourf(
    x_lin, y_lin, Z_log,
    levels=50,
    cmap="viridis",
    norm=LogNorm(vmin=Z_log.min(), vmax=Z_log.max())
)
cs_lines = plt.contour(
    x_lin, y_lin, Z_log,
    levels=50,
    colors='white',
    linewidths=0.3,
    norm=LogNorm(vmin=Z_log.min(), vmax=Z_log.max())
)
plt.colorbar(cs, label=r"$F_{\mathrm{true}}(x)$ (log scale)")

plt.scatter(x_min_true[0], x_min_true[1],
            c="white", marker="o", s=200,
            edgecolors="black", linewidths=1.5,
            label=r"Global Min of $F_{\mathrm{true}}$")

for idx in sorted_indices:
    all_found_x = results[idx]["all_found_x"]
    x_vals = [p[0] for p in all_found_x]
    y_vals = [p[1] for p in all_found_x]
    plt.scatter(
        x_vals, y_vals, marker="X", s=50, edgecolors="black",
        label=rf"Condition {idx+1} minima"
    )

plt.title(r"Contour of $F_{\mathrm{true}}$")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.legend()
plt.tight_layout()
plt.savefig("contour_F_true_all_solutions.pdf", dpi=300)
plt.close()

print("A new log-scale contour plot of F_true with all solutions has been generated!")
