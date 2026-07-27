import pandas as pd
import numpy as np
import json
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Data Loading ---
with open("data/fatigue_data.json", encoding="utf-8") as f:
    dataset = json.load(f)
records = dataset["data_points"]
df = pd.DataFrame(records)

print(f"Initial data loaded with {len(df)} samples.")

# Column selection based on schema and request context
X_col = 'ct_sv'
Y_col = 'cycles'

x_data = np.array(df[X_col], dtype=float)
y_data = np.array(df[Y_col], dtype=float)

print(f"Data preparation complete. Using {len(x_data)} samples for analysis.")

# --- Simulation Parameters ---
N_ITERATIONS = 30
NOISE_SIGMA = 2.0
np.random.seed(42) # For reproducibility of synthetic data generation

# Storage for results
mean_bandwidths = []
all_final_datasets = []

# --- Main Experiment Loop ---
for i in range(N_ITERATIONS):
    # 1. Generate Synthetic Data (10 points)
    # Sample along the fitted curve using the original data's x-range for sampling locations
    x_sample_locations = np.linspace(x_data.min(), x_data.max(), 10)
    
    # Fit a temporary polynomial to estimate the underlying curve shape for noise addition
    temp_coeffs = np.polyfit(x_data, y_data, min(2, len(x_data) - 1))
    base_curve_estimate = np.polyval(temp_coeffs, x_sample_locations)
    
    # Generate noisy points
    noise = np.random.normal(0, NOISE_SIGMA, size=len(x_sample_locations))
    synthetic_y = base_curve_estimate + noise
    synthetic_x = x_sample_locations
    
    # 2. Combine Original and Synthetic Data (Total 11 + 10 = 21 points)
    X_combined = np.concatenate([x_data, synthetic_x])
    Y_combined = np.concatenate([y_data, synthetic_y])
    
    # 3. Polynomial Fitting using AIC
    
    # Log transform Y for fitting (since cycles/counts must be positive)
    log_y_combined = np.log(Y_combined)
    
    best_degree = -1
    min_aic = float('inf')
    final_coeffs = None
    
    max_degree = max(1, len(X_combined) // 4)
    degrees_to_test = range(1, min(3, max_degree + 1)) # Test up to degree 2 or max allowed

    aic_results = {}
    print(f"\n--- Iteration {i+1}/{N_ITERATIONS}: Fitting ---")
    for degree in degrees_to_test:
        # Fit on log scale
        coeffs = np.polyfit(X_combined, log_y_combined, degree)
        log_y_hat = np.polyval(coeffs, X_combined)
        rss = np.sum((log_y_combined - log_y_hat) ** 2)
        # AIC formula: n * log(RSS/n) + 2k (where k is number of parameters = degree + 1)
        aic = len(X_combined) * np.log(rss / len(X_combined)) + 2 * (degree + 1)
        aic_results[degree] = aic

    # Select best degree based on AIC
    best_degree = min(aic_results, key=aic_results.get)
    min_aic = aic_results[best_degree]
    final_coeffs = np.polyfit(X_combined, log_y_combined, best_degree)

    print(f"AIC Results: { {d: f'{a:.2f}' for d, a in aic_results.items()} }")
    print(f"Selected Degree (Lowest AIC): {best_degree} with AIC = {min_aic:.2f}")
    print(f"Fitted Coefficients: {final_coeffs}")

    # 4. Residual Bootstrap Uncertainty Band Calculation
    
    # Fit on the combined data again to get base residuals for this iteration
    base_at_x = np.polyval(final_coeffs, X_combined)
    residuals = log_y_combined - base_at_x # shape (n,) — log space residuals
    
    x_range = np.linspace(X_combined.min(), X_combined.max(), 200)
    boot_preds = np.zeros((500, 200))
    rng = np.random.default_rng(seed=42 + i) # Seed based on iteration for slight variation
    
    for j in range(500):
        resampled = rng.choice(residuals, size=len(X_combined), replace=True)
        log_y_boot = base_at_x + resampled  # shape (n,) — add to base at original x
        c = np.polyfit(X_combined, log_y_boot, best_degree)
        boot_preds[j] = np.polyval(c, x_range) # shape (200,) — predict on grid

    # Calculate 94% confidence band (3rd and 97th percentiles)
    lower = np.clip(np.exp(np.percentile(boot_preds, 3, axis=0)), 0, None)
    upper = np.clip(np.exp(np.percentile(boot_preds, 97, axis=0)), 0, y_data.max() * 2)
    mean_fit = np.exp(np.polyval(final_coeffs, x_range))

    # Calculate mean band width (Mean of upper - lower across the range)
    bandwidth = np.mean(upper - lower)
    mean_bandwidths.append(bandwidth)
    
    # Store data for final plot visualization
    all_final_datasets.append({
        'X': X_combined, 
        'Y': Y_combined, 
        'lower': lower, 
        'upper': upper, 
        'mean': mean_fit, 
        'x_range': x_range
    })

print("\n--- Experiment Complete ---")

# --- Plotting Results ---

# (1) Mean band width vs iteration number showing convergence
plt.figure(figsize=(10, 6))
plt.plot(np.arange(N_ITERATIONS), mean_bandwidths, marker='o', linestyle='-', label="Mean Band Width")
plt.title("Convergence of Mean Uncertainty Band Width over Iterations")
plt.xlabel("Iteration Number")
plt.ylabel("Mean Band Width (Upper - Lower)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.savefig("./results/bandwidth_convergence.png", dpi=150, bbox_inches="tight")
print("Generated Plot 1: Mean band width vs iteration number saved to ./results/bandwidth_convergence.png")


# (2) The final dataset with fitted curve and uncertainty band (Using the last iteration's results)
final_data = all_final_datasets[-1]

plt.figure(figsize=(10, 6))

# Scatter plot of original data points
plt.scatter(x_data, y_data, color="steelblue", zorder=5, label="Original Data")

# Fill between the uncertainty band (using the last iteration's calculation)
plt.fill_between(final_data['x_range'], final_data['lower'], final_data['upper'], alpha=0.3, color="crimson", label="94% Band (Bootstrap)")

# Plot mean fit curve
plt.plot(final_data['x_range'], final_data['mean'], color="blue", linewidth=2, label=f"Fit Curve (Degree={best_degree})")

# Add legend and grid
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Set labels according to schema/request
plt.xlabel("CT Sv (um)")
plt.ylabel("Number of Cycles to Failure")
plt.title("Final Fit Result: Polynomial Curve and Uncertainty Band")

plt.savefig("./results/final_fit_plot.png", dpi=150, bbox_inches="tight")
print("Generated Plot 2: Final dataset fit curve and uncertainty band saved to ./results/final_fit_plot.png")

# --- Summary Output ---
print("\n--- Analysis Summary ---")
print(f"Mean Band Widths recorded over {N_ITERATIONS} iterations.")
print(f"Average Mean Band Width: {np.mean(mean_bandwidths):.2f}")
print("Script finished successfully, generating required plots.")