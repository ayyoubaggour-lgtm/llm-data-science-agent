import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# --- Data Loading ---
with open("data/fatigue_data.json", encoding="utf-8") as f:
    dataset = json.load(f)
records = dataset["data_points"]
x = np.array([r["ct_sv"] for r in records], dtype=float)
y = np.array([r["cycles"] for r in records], dtype=float)

# --- Polynomial Fitting with AIC ---
log_y = np.log(y)
aic_results = {}
for degree in [1, 2]:
    coeffs = np.polyfit(x, log_y, degree)
    log_y_hat = np.polyval(coeffs, x)
    rss = np.sum((log_y - log_y_hat) ** 2)
    aic = len(x) * np.log(rss / len(x)) + 2 * (degree + 1)
    print(f"  degree={degree}  AIC={aic:.2f}")
    aic_results[degree] = aic

# Determine best degree based on AIC and cap constraint
max_allowed_degree = max(1, len(x) // 4)
best_degree = -1
min_aic = float('inf')

for degree in [1, 2]:
    if degree <= max_allowed_degree:
        if aic_results[degree] < min_aic:
            min_aic = aic_results[degree]
            best_degree = degree

# Re-calculate coefficients for the best degree found
coeffs = np.polyfit(x, log_y, best_degree)

print("\n--- Results ---")
print(f"Selected Degree: {best_degree}")
print(f"Minimum AIC: {min_aic:.2f}")
print("Fitted Coefficients:", coeffs)

# Calculate R-squared for the best fit (optional key numeric result)
log_y_hat = np.polyval(coeffs, x)
ss_total = np.sum((log_y - np.mean(log_y)) ** 2)
ss_residual = np.sum((log_y - log_y_hat) ** 2)
r_squared = 1 - (ss_residual / ss_total)
print(f"R-squared: {r_squared:.4f}")


# --- Residual Bootstrap Uncertainty Band ---
base_at_x  = np.polyval(coeffs, x)          # shape (n,) — fit at original x points
residuals  = log_y - base_at_x              # shape (n,) — log space residuals
x_range    = np.linspace(x.min(), x.max(), 200)
boot_preds = np.zeros((500, 200))
rng        = np.random.default_rng(seed=42)

for i in range(500):
    resampled     = rng.choice(residuals, size=len(x), replace=True)
    log_y_boot    = base_at_x + resampled  # shape (n,) — add to base at original x
    c             = np.polyfit(x, log_y_boot, best_degree)
    boot_preds[i] = np.polyval(c, x_range)   # shape (200,) — predict on grid

lower = np.clip(np.exp(np.percentile(boot_preds, 3,  axis=0)), 0, None)
upper = np.clip(np.exp(np.percentile(boot_preds, 97, axis=0)), 0, y.max() * 2)
mean  = np.exp(np.polyval(coeffs, x_range))


# --- Plotting ---
plt.figure(figsize=(10, 6))

# Data points
plt.scatter(x, y, color="steelblue", zorder=5, label="Data")

# Uncertainty band (94%)
plt.fill_between(x_range, lower, upper, alpha=0.3, color="crimson", label="94% band")

# Mean fit curve
plt.plot(x_range, mean, color="blue", linewidth=2, label=f"Fit (degree={best_degree})")

# Labels and Title based on schema/request
plt.xlabel("CT Sv (um)")
plt.ylabel("Number of Cycles to Failure")
plt.title("CT Sv (defect-derived stress metric, micrometers) vs cycles to failure")

plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Save the plot
os.makedirs("./results", exist_ok=True)
plt.savefig("./results/output_plot.png", dpi=150, bbox_inches="tight")
plt.close()