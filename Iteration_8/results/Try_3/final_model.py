import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score

# --- Data Loading ---
# Ensure the data directory exists (for standalone execution)
os.makedirs("./results", exist_ok=True)

with open("data/fatigue_data.json", encoding="utf-8") as f:
    dataset = json.load(f)
records = dataset["data_points"]
x = np.array([r["ct_sv"] for r in records], dtype=float)
y = np.array([r["cycles"] for r in records], dtype=float)

# --- Preprocessing and Model Selection ---
log_y = np.log(y)
n_samples = len(x)
max_degree_allowed = max(1, n_samples // 4)
degrees_to_test = [d for d in [1, 2] if d <= max_degree_allowed]

aic_results = {}
best_degree = -1
min_aic = np.inf

print("--- Model Selection using AIC ---")

for degree in degrees_to_test:
    # Fit polynomial on log scale
    coeffs = np.polyfit(x, log_y, degree)
    log_y_hat = np.polyval(coeffs, x)
    
    # Calculate Residual Sum of Squares (RSS)
    rss = np.sum((log_y - log_y_hat) ** 2)
    
    # AIC calculation: N * ln(RSS/N) + 2k
    aic = n_samples * np.log(rss / n_samples) + 2 * (degree + 1)
    
    aic_results[degree] = aic
    print(f"  degree={degree}  AIC={aic:.2f}")

# Select best degree
if aic_results:
    best_degree = min(aic_results, key=aic_results.get)
    min_aic = aic_results[best_degree]
else:
    print("Error: No degrees could be tested.")
    exit()

# --- Final Fit and Coefficients ---
coeffs = np.polyfit(x, log_y, best_degree)
base_at_x = np.polyval(coeffs, x)

# Calculate R^2 for the selected model (on log scale residuals vs predicted log values)
r2 = r2_score(log_y, base_at_x)

print("\n--- Model Summary ---")
print(f"Selected Degree: {best_degree} (Lowest AIC: {min_aic:.2f})")
print(f"R-squared on log scale: {r2:.4f}")
print("Fitted Coefficients:", coeffs)


# --- Uncertainty Band Bootstrap ---
rng = np.random.default_rng(seed=42)
n_bootstrap = 500

x_range = np.linspace(x.min(), x.max(), 200)
boot_preds = np.zeros((n_bootstrap, 200))

for i in range(n_bootstrap):
    # Resample residuals (shape: n_samples,)
    resampled = rng.choice(base_at_x - log_y, size=n_samples, replace=True) # Note: Residuals are calculated as log_y - base_at_x in the prompt description, but for bootstrap we use the difference (log_y - base_at_x). Let's stick to the definition provided: residuals = log_y - base_at_x
    resampled = rng.choice(base_at_x - log_y, size=n_samples, replace=True) # Correcting residual calculation based on standard practice (observed - predicted)

# Re-reading rule: residuals  = log_y - base_at_x
residuals = log_y - base_at_x 

for i in range(n_bootstrap):
    resampled = rng.choice(residuals, size=n_samples, replace=True)
    log_y_boot = base_at_x + resampled  # shape (n,) — add to base at original x
    c = np.polyfit(x, log_y_boot, best_degree)
    boot_preds[i] = np.polyval(c, x_range)   # shape (200,)

lower = np.clip(np.exp(np.percentile(boot_preds, 3, axis=0)), 0, None)
upper = np.clip(np.exp(np.percentile(boot_preds, 97, axis=0)), 0, y.max() * 2)
mean  = np.exp(np.polyval(coeffs, x_range))


# --- Plotting ---
plt.figure(figsize=(10, 6))

# Scatter data (Original scale)
plt.scatter(x, y, color="steelblue", zorder=5, label="Data")

# Uncertainty Band (Original scale)
plt.fill_between(x_range, lower, upper, alpha=0.3, color="crimson", label="94% band")

# Mean Fit Curve (Original scale)
plt.plot(x_range, mean, color="blue", linewidth=2, label=f"Fit (degree={best_degree})")

# Plot formatting
plt.xlabel("CT Sv (um)")
plt.ylabel("Number of Cycles to Failure")
plt.title("Fatigue Life Prediction using Polynomial Regression")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Save the plot
plt.savefig("./results/output_plot.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSuccessfully generated and saved the prediction plot to ./results/output_plot.png.")