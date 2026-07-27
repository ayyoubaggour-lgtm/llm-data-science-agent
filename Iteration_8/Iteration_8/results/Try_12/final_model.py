import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# --- Data Loading ---
with open("data/fatigue_data.json", encoding="utf-8") as f:
    dataset = json.load(f)
records = dataset["data_points"]
df = pd.DataFrame(records)

print(f"Initial data loaded with {len(df)} samples.")

# Column selection based on schema and request context (x='ct_sv', y='cycles')
X_col = 'ct_sv'
Y_col = 'cycles'

# Select relevant columns and cast to float numpy arrays
try:
    x_data = np.array(df[X_col], dtype=float)
    y_data = np.array(df[Y_col], dtype=float)
except KeyError as e:
    print(f"Error: Required column {e} not found in the dataset.")
    exit()

# --- Simulation Parameters ---
N_ITERATIONS = 30
NOISE_SIGMA = 5000.0

# Store results for plotting convergence
mean_bandwidths = []

# --- Main Simulation Loop ---
for i in range(N_ITERATIONS):
    # 1. Generate synthetic data points
    # Sample along the fitted curve (using original data trend as guide)
    x_sampled = np.linspace(x_data.min(), x_data.max(), 10)
    # Fit a temporary polynomial to estimate the underlying curve for sampling
    temp_coeffs = np.polyfit(x_data, y_data, min(2, len(x_data) - 1))
    y_curve_estimate = np.polyval(temp_coeffs, x_sampled)

    # Generate noise and synthetic points (10 new points)
    noise = np.random.normal(0, NOISE_SIGMA, size=10)
    synthetic_y = y_curve_estimate[:10] + noise
    synthetic_x = x_sampled[np.random.randint(0, 10, 10)] # Use random X values from the sampled set

    # Combine original and synthetic data
    all_x = np.concatenate([x_data, synthetic_x])
    all_y = np.concatenate([y_data, synthetic_y])

    # 2. Fit best polynomial on all 21 points using AIC
    n_points = len(all_x)
    max_degree = max(1, n_points // 4)
    best_degree = 1
    min_aic = float('inf')
    final_coeffs = None

    # Fit in log space as Y represents cycles (counts/quantities that must be positive)
    log_y = np.log(all_y)

    for degree in range(1, min(max_degree + 1, 3)): # Test up to degree=2 for stability, capped by max_degree
        if degree > len(all_x): break
        try:
            coeffs = np.polyfit(all_x, log_y, degree)
            log_y_hat = np.polyval(coeffs, all_x)
            rss = np.sum((log_y - log_y_hat) ** 2)
            # AIC formula: n * log(RSS/n) + 2k (where k is number of parameters = degree + 1)
            aic = len(all_x) * np.log(rss / len(all_x)) + 2 * (degree + 1)

            print(f"Iteration {i+1}/{N_ITERATIONS} - Testing degree={degree}: AIC={aic:.2f}")

            if aic < min_aic:
                min_aic = aic
                best_degree = degree
                final_coeffs = coeffs
        except Exception as e:
            print(f"Could not fit polynomial for degree {degree} in iteration {i+1}: {e}")
            continue

    # 3. Compute residual bootstrap uncertainty band (using the best model found)
    if final_coeffs is None or best_degree == 0:
        print(f"Iteration {i+1}: Failed to determine a stable polynomial fit.")
        mean_bandwidths.append(np.nan)
        continue

    # Re-calculate base fit and residuals using the chosen degree/coefficients
    base_at_x = np.polyval(final_coeffs, all_x)
    residuals = log_y - base_at_x # shape (n,) — log space residuals

    # Bootstrap setup
    rng = np.random.default_rng(seed=42 + i) # Seed changes per iteration for independent runs
    boot_preds = np.zeros((500, 200))
    x_range = np.linspace(all_x.min(), all_x.max(), 200)

    for j in range(500):
        resampled = rng.choice(residuals, size=len(all_x), replace=True)
        log_y_boot = base_at_x + resampled # shape (n,) — add to base at original x
        c = np.polyfit(all_x, log_y_boot, best_degree)
        boot_preds[j] = np.polyval(c, x_range)

    # Calculate band limits and mean prediction
    lower = np.clip(np.exp(np.percentile(boot_preds, 3, axis=0)), 0, None)
    upper = np.clip(np.exp(np.percentile(boot_preds, 97, axis=0)), 0, y_data.max() * 2) # Cap upper bound reasonably
    mean_pred = np.exp(np.polyval(final_coeffs, x_range))

    # Calculate mean band width (Mean of Upper - Lower) across the range
    bandwidth = np.mean(upper - lower)
    mean_bandwidths.append(bandwidth)

print("\n--- Simulation Complete ---")
print(f"Average Mean Bandwidth over {N_ITERATIONS} iterations: {np.nanmean(mean_bandwidths):.2f}")


# 4. Plotting Results

# (1) mean band width vs iteration number showing convergence
plt.figure()
# FIX: Use the loop index 'i' to correctly plot against the list length, not a variable that might be out of scope/incorrectly scoped in the original error context.
x_values = np.arange(1, N_ITERATIONS + 1)
y_values = mean_bandwidths[:N_ITERATIONS] # Ensure we only take N_ITERATIONS values
plt.plot(x_values, y_values, marker='o', linestyle='-', label="Mean Bandwidth")
plt.xlabel("Iteration Number")
plt.ylabel("Average Mean Band Width (Upper - Lower)")
plt.title("Convergence of Mean Bootstrap Band Width")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("./results/convergence_bandwidth.png", dpi=150, bbox_inches="tight")
plt.close()

# (2) The final dataset with fitted curve and uncertainty band (using the last iteration's results for visualization consistency)
final_x = all_x
final_y = all_y
final_coeffs_plot = final_coeffs
best_degree_plot = best_degree
mean_pred_plot = np.exp(np.polyval(final_coeffs_plot, np.linspace(final_x.min(), final_x.max(), 200)))
lower_plot = lower
upper_plot = upper

plt.figure()
# Plotting elements as required: Data, Band, Mean Fit Curve
plt.scatter(x_data, y_data, color="steelblue", zorder=5, label="Original Data")
plt.scatter(synthetic_x, synthetic_y, color="darkorange", alpha=0.6, zorder=4, label="Synthetic Noise Points (Last Iter)")

# Fill band using the 200-point range used for calculation
plt.fill_between(x_range, lower_plot, upper_plot, alpha=0.3, color="crimson", label="94% Band")
# Plot mean fit curve
plt.plot(x_range, mean_pred_plot, color="blue", linewidth=2, label=f"Fit (degree={best_degree_plot})")

plt.xlabel("CT Sv (um)")
plt.ylabel("Number of Cycles to Failure")
plt.title("Final Fit and Uncertainty Band on Combined Dataset")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("./results/final_fit_uncertainty_band.png", dpi=150, bbox_inches="tight")

print("\nScript finished successfully. Two plots saved to ./results/")