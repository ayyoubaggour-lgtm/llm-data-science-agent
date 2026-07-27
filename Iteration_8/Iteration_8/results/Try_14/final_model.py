import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from scipy import stats

# --- Data Loading ---
with open("data/fatigue_data.json", encoding="utf-8") as f:
    dataset = json.load(f)
records = dataset["data_points"]
df = pd.DataFrame(records)

print(f"Starting analysis with {len(df)} initial data points.")

# Column Selection based on schema and request context
# 'ct_sv' (numeric): sample values = [98, 96, 81, 78, 70] -> Likely X
# 'cycles' (numeric): sample values = [20000, 30000, 22000, 32000, 38000] -> Likely Y
X_COL = 'ct_sv'
Y_COL = 'cycles'

# Select and cast data
x_data = np.array(df[X_COL], dtype=float)
y_data = np.array(df[Y_COL], dtype=float)

print(f"Data loaded: X={X_COL} ({len(x_data)} points), Y={Y_COL} ({len(y_data)} points)")

# --- Simulation Parameters ---
N_ITERATIONS = 30
SIGMA_NOISE = 1.0

# --- Core Analysis Variables Initialization ---
all_band_widths = []
cumulative_means = []

# --- Iteration Loop ---
for i in range(N_ITERATIONS):
    # 1. Generate synthetic points
    # Sample along the fitted curve (using original data for initial fit)
    # We use the current best estimate of the function to sample from, but since this is the first iteration, we use the raw data trend.
    # For simplicity and stability across iterations, we'll generate noise based on a linear interpolation/fit of the *original* data range.

    # Determine sampling points for synthetic data (e.g., 10 evenly spaced points between min/max X)
    x_synth_base = np.linspace(x_data.min(), x_data.max(), 10)
    
    # Fit a temporary line to the original data to estimate the curve height for noise addition
    coeffs_temp = np.polyfit(x_data, y_data, 1)
    y_synth_base = np.polyval(coeffs_temp, x_synth_base)

    # Generate Gaussian noise and synthetic points
    noise = np.random.normal(0, SIGMA_NOISE, size=10)
    x_synthetic = x_synth_base
    y_synthetic = y_synth_base + noise

    # 2. Combine original and synthetic data
    X_combined = np.concatenate([x_data, x_synthetic])
    Y_combined = np.concatenate([y_data, y_synthetic])
    N_total = len(X_combined)

    # 3. Polynomial Fitting using AIC (Log Space for Y)
    log_y_combined = np.log(Y_combined)
    
    best_degree = -1
    min_aic = float('inf')
    final_coeffs = None
    
    max_degree_to_test = max(1, N_total // 4)
    degrees_to_test = list(range(1, min(3, max_degree_to_test) + 1)) # Test up to degree 2 or limited by data size

    aic_results = {}
    for degree in degrees_to_test:
        # Fit on log-transformed data
        coeffs = np.polyfit(X_combined, log_y_combined, degree)
        log_y_hat = np.polyval(coeffs, X_combined)
        rss = np.sum((log_y_combined - log_y_hat) ** 2)
        # AIC formula: n * log(RSS/n) + 2k (where k is number of parameters = degree + 1)
        aic = N_total * np.log(rss / N_total) + 2 * (degree + 1)
        aic_results[degree] = aic

    # Select best degree based on AIC
    if aic_results:
        best_degree = min(aic_results, key=aic_results.get)
        min_aic = aic_results[best_degree]
        final_coeffs = np.polyfit(X_combined, log_y_combined, best_degree)
    else:
        print("Warning: Could not determine polynomial degree.")
        best_degree = 1 # Fallback

    # --- Residual Bootstrap Uncertainty Band Calculation ---
    base_at_x = np.polyval(final_coeffs, X_combined)
    residuals = log_y_combined - base_at_x  # Shape (N_total,)
    
    x_range = np.linspace(X_combined.min(), X_combined.max(), 200)
    boot_preds = np.zeros((500, 200))
    rng = np.random.default_rng(seed=42 + i) # Seed changes per iteration for independent runs
    
    for j in range(500):
        resampled = rng.choice(residuals, size=N_total, replace=True)
        log_y_boot = base_at_x + resampled  # Add to base at original x points
        c = np.polyfit(X_combined, log_y_boot, best_degree)
        boot_preds[j] = np.polyval(c, x_range) # Predict on grid

    lower = np.clip(np.exp(np.percentile(boot_preds, 3, axis=0)), 0, None)
    upper = np.clip(np.exp(np.percentile(boot_preds, 97, axis=0)), 0, y_data.max() * 2) # Use original Y max for upper bound cap
    mean_fit = np.exp(np.polyval(final_coeffs, x_range))

    # Calculate band width (Mean of Upper - Lower)
    current_band_width = np.mean(upper - lower)
    all_band_widths.append(current_band_width)

    # 4. Compute Cumulative Mean
    cumulative_means.append(np.mean(all_band_widths[:i+1]))

    # --- Output for current iteration ---
    print(f"\n--- Iteration {i+1}/{N_ITERATIONS} ---")
    print(f"Raw Band Width: {current_band_width:.4f}")
    print(f"Cumulative Mean Band Width: {cumulative_means[-1]:.4f}")

# --- Plotting Results ---

# 1. Cumulative Mean vs Iteration Number
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
iterations = np.arange(1, N_ITERATIONS + 1)
plt.plot(iterations, cumulative_means, marker='o', linestyle='-', color='purple')
plt.xlabel("Iteration Number")
plt.ylabel("Cumulative Mean Band Width (Mean of Upper - Lower)")
plt.title("Convergence of Cumulative Mean Band Width")
plt.grid(True, linestyle="--", alpha=0.6)

# 2. Final Dataset Plot (Last Iteration's combined points)
plt.subplot(1, 2, 2)
plt.scatter(X_combined, Y_combined, color="steelblue", zorder=5, label="Combined Data (Original + Synthetic)")
plt.fill_between(x_range, lower, upper, alpha=0.3, color="crimson", label="94% band")
plt.plot(x_range, mean_fit, color="blue", linewidth=2, label=f"Fit (degree={best_degree})")

# Add original data points explicitly for clarity on the plot
plt.scatter(x_data, y_data, color="darkorange", zorder=6, label="Original Data Points")

plt.xlabel("CT Sv (um)")
plt.ylabel("Number of Cycles to Failure")
plt.title(f"Final Fit and Uncertainty Band (Degree={best_degree})")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Save plots
plt.savefig("./results/output_plot.png", dpi=150, bbox_inches="tight")

print("\n=========================================================")
print("Analysis Complete.")
print(f"The script successfully ran {N_ITERATIONS} iterations.")
print(f"Results saved to ./results/output_plot.png")
print("---------------------------------------------------------")
print("Summary of Final Values:")
print(f"Final Raw Band Width (Iteration {N_ITERATIONS}): {all_band_widths[-1]:.4f}")
print(f"Final Cumulative Mean Band Width: {cumulative_means[-1]:.4f}")