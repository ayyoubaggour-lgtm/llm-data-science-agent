import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from sklearn.metrics import r2_score

# Data Loading
with open("data/fatigue_data.json", encoding="utf-8") as f:
    dataset = json.load(f)
records = dataset["data_points"]
df = pd.DataFrame(records)

# Column Selection and Preparation
X = np.array(df["ct_sv"], dtype=float)
Y = np.array(df["cycles"], dtype=float)

N = len(X)
max_degree = max(1, N // 4) # For N=11, max_degree = 2

# --- Polynomial Fitting and AIC Selection ---
aic_results = {}
best_degree = 0
min_aic = np.inf
coeffs = None

print("--- Model Selection using AIC ---")

for degree in range(1, max_degree + 1):
    log_y = np.log(Y)
    
    # Fit polynomial
    try:
        coeffs_temp = np.polyfit(X, log_y, degree)
        
        # Calculate predicted values and RSS
        log_y_hat = np.polyval(coeffs_temp, X)
        rss = np.sum((log_y - log_y_hat) ** 2)
        
        # AIC calculation: N * log(RSS/N) + 2k (where k is number of parameters = degree + 1)
        aic = N * np.log(rss / N) + 2 * (degree + 1)
        aic_results[degree] = aic
        print(f"  degree={degree}  AIC={aic:.2f}")

    except Exception as e:
        # Handle potential fitting errors, though unlikely here
        print(f"Error fitting degree {degree}: {e}")
        continue

if aic_results:
    best_degree = min(aic_results, key=aic_results.get)
    min_aic = aic_results[best_degree]
    coeffs = np.polyfit(X, np.log(Y), best_degree)
else:
    print("Could not determine the best model degree.")

# --- Uncertainty Band Calculation (Residual Bootstrap) ---
if coeffs is not None and best_degree > 0:
    base_at_x = np.polyval(coeffs, X)          # shape (n,) — fit at original x points
    residuals  = np.log(Y) - base_at_x        # shape (n,) — log space residuals
    
    x_range = np.linspace(X.min(), X.max(), 200)
    boot_preds_grid = np.zeros((500, 200)) # For plotting the band over x_range
    boot_preds_x = np.zeros((500, N))     # For checking data points at X

    rng = np.random.default_rng(seed=42)

    for i in range(500):
        resampled     = rng.choice(residuals, size=N, replace=True)
        log_y_boot    = base_at_x + resampled  # shape (n,) — add to base at original x
        c             = np.polyfit(X, log_y_boot, best_degree)
        
        # Predict on the full grid for plotting the band
        boot_preds_grid[i] = np.polyval(c, x_range)   # shape (200,) 
        
        # Predict at original data points X for coloring check
        boot_preds_x[i] = np.polyval(c, X)           # shape (N,)

    # Calculate bounds over the full range (for plotting fill_between)
    lower = np.clip(np.exp(np.percentile(boot_preds_grid, 3,  axis=0)), 0, None)
    upper = np.clip(np.exp(np.percentile(boot_preds_grid, 97, axis=0)), 0, Y.max() * 2)

    # Calculate bounds at original data points X (for coloring scatter dots)
    lower_at_x = np.clip(np.exp(np.percentile(boot_preds_x, 3,  axis=0)), 0, None) # shape (N,)
    upper_at_x = np.clip(np.exp(np.percentile(boot_preds_x, 97, axis=0)), 0, Y.max() * 2) # shape (N,)

    mean  = np.exp(np.polyval(coeffs, x_range))

# --- Plotting and Visualization ---
plt.figure(figsize=(10, 6))

# Identify points outside the uncertainty band using bounds calculated at X
is_outside = (Y < lower_at_x) | (Y > upper_at_x)

# Plotting data points: inside/normal color vs outside/different color
plt.scatter(X[~is_outside], Y[~is_outside], color="steelblue", zorder=5, label="Data (Within 95% CI)")
plt.scatter(X[is_outside], Y[is_outside], color="red", zorder=6, label="Data (Outside 95% CI)")

# Plot uncertainty band (using bounds calculated over x_range)
plt.fill_between(x_range, lower, upper, alpha=0.3, color="crimson", label="94% Confidence Band")

# Plot fitted curve
plt.plot(x_range, mean, color="blue", linewidth=2, label=f"Fit (degree={best_degree})")

# Add plot elements
plt.xlabel("CT Sv (um)")
plt.ylabel("Number of Cycles to Failure")
plt.title("Fatigue Life Prediction using Polynomial Regression")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Save the figure
plt.savefig("./results/output_plot.png", dpi=150, bbox_inches="tight")

# --- Summary Output ---
print("\n--- Analysis Results ---")
if best_degree > 0:
    print(f"Best Model Degree selected by AIC: {best_degree} (AIC={min_aic:.2f})")
    print("Fitted Coefficients:", coeffs)
    
    # Calculate R² for the fit in log space, then interpret it.
    log_y_pred = np.polyval(coeffs, X)
    r_squared = r2_score(np.log(Y), log_y_pred)
    print(f"R-squared (Log Space): {r_squared:.4f}")

    # Print summary of the process
    print("\nSuccessfully fitted polynomial model and generated uncertainty band plot.")
else:
    print("Model fitting failed or was skipped.")