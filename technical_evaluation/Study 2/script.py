import numpy as np
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

# ------------------------------------------------------
# 1. DEFINE SYNTHETIC FUNCTIONS
# ------------------------------------------------------
np.random.seed(42)

def true_function(x):
    return 0.5 * (np.sin(x) + np.cos(x))

def source1(x):  return 1.0*np.sin(x) + 0.0*np.cos(x)
def source2(x):  return 0.2*np.sin(x) + 0.8*np.cos(x)
def source3(x):  return 0.4*np.sin(x) + 0.6*np.cos(x)

# ------------------------------------------------------
# 2. SAMPLE DATA *SPARSELY* IN DIFFERENT REGIONS
# ------------------------------------------------------
# Full domain we want to test on:
x_min, x_max = 0, 6

# Source1 covers the entire range but with moderate density
X_s1 = np.linspace(x_min, x_max, 15)[:, None]
noise_s1 = 0.15
Y_s1 = source1(X_s1) + noise_s1 * np.random.randn(*X_s1.shape)

# Source2 covers only [0..3]
X_s2 = np.linspace(x_min, 3, 8)[:, None]
noise_s2 = 0.10
Y_s2 = source2(X_s2) + noise_s2 * np.random.randn(*X_s2.shape)

# Source3 covers only [3..6]
X_s3 = np.linspace(3, x_max, 8)[:, None]
noise_s3 = 0.05
Y_s3 = source3(X_s3) + noise_s3 * np.random.randn(*X_s3.shape)

# ------------------------------------------------------
# 3. TRAIN A GP FOR EACH SOURCE
# ------------------------------------------------------
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)

gp_s1 = GaussianProcessRegressor(kernel=kernel, alpha=noise_s1**2, normalize_y=True)
gp_s2 = GaussianProcessRegressor(kernel=kernel, alpha=noise_s2**2, normalize_y=True)
gp_s3 = GaussianProcessRegressor(kernel=kernel, alpha=noise_s1**2, normalize_y=True)

gp_s1.fit(X_s1, Y_s1.ravel())
gp_s2.fit(X_s2, Y_s2.ravel())
gp_s3.fit(X_s3, Y_s3.ravel())

# ------------------------------------------------------
# 4. PREDICT & FORM "COMBINED GP"
# ------------------------------------------------------
Xtest = np.linspace(x_min, x_max, 200)[:, None]
f_true = true_function(Xtest)

mean_s1, std_s1 = gp_s1.predict(Xtest, return_std=True)
mean_s2, std_s2 = gp_s2.predict(Xtest, return_std=True)
mean_s3, std_s3 = gp_s3.predict(Xtest, return_std=True)

# Combined (naive) mean & std: average of each
combined_mean = (mean_s1 + mean_s2 + mean_s3) / 3.0
combined_std  = (std_s1  + std_s2  + std_s3)  / 3.0

# ------------------------------------------------------
# 5. DISAGREEMENT & AVERAGE VARIANCE
# ------------------------------------------------------
# Disagreement = var across GP means
all_means    = np.vstack([mean_s1, mean_s2, mean_s3])   # shape: (3, 200)
disagreement = np.var(all_means, axis=0)               # shape: (200,)

# Average predictive variance (using stds directly for shading)
all_stds = np.vstack([std_s1**2, std_s2**2, std_s3**2])          # shape: (3, 200)
avg_var  = np.mean(all_stds, axis=0)                   # shape: (200,)

lower_shade = disagreement - avg_var
upper_shade = disagreement + avg_var

# ------------------------------------------------------
# 6. PLOT: TWO SUBPLOTS
# ------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 8), sharex=True)

# --- Top subplot ---
ax1.set_title("Multi-Surrogate GP: Individual Surrogates and Combined Prediction")
ax1.scatter(X_s1, Y_s1, marker='x', label=r"$S_1$ samples")
ax1.plot(Xtest, mean_s1, label=r"$S_1$ GP")

ax1.scatter(X_s2, Y_s2, marker='x', color='green', label=r"$S_2$ samples")
ax1.plot(Xtest, mean_s2, color='green', label=r"$S_2$ GP")

ax1.scatter(X_s3, Y_s3, marker='x', color='orange', label=r"$S_3$ samples")
ax1.plot(Xtest, mean_s3, color='orange', label=r"$S_3$ GP")

ax1.plot(Xtest, f_true, 'k--', label="True function")

# Combined GP
ax1.plot(Xtest, combined_mean, 'r', label="Multi-Surrogate GP")
ax1.fill_between(
    Xtest.ravel(),
    combined_mean - combined_std,
    combined_mean + combined_std,
    color='red', alpha=0.2,
    label="Combined Uncertainty"
)
ax1.legend(loc='upper right')
ax1.set_ylabel("Function Value")

# --- Bottom subplot ---
#ax2.set_title("Disagreement = Var of Means, with ± Average Std Shading")
ax2.set_title("Disagreement Between Sources")
ax2.plot(Xtest, disagreement, 'r', label="Disagreement")
ax2.fill_between(
    Xtest.ravel(),
    lower_shade,
    upper_shade,
    color='gray', alpha=0.2,
    label="± Average GP Std"
)
ax2.set_xlabel("X")
ax2.set_ylabel("Disagreement")
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig("sparse_regions_increase_uncertainty.pdf", dpi=300)
plt.close(fig)
