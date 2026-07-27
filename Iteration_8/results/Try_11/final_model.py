import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# --- Data Loading ---
file_path = "data/fatigue_data.json"
with open(file_path, encoding="utf-8") as f:
    dataset = json.load(f)
records = dataset["data_points"]
df = pd.DataFrame(records)

# Column Selection based on schema and request context (x='ct_sv', y='cycles')
X_col = 'ct_sv'
Y_col = 'cycles'

# Select relevant columns and ensure correct types
try:
    df_selected = df[[X_col, Y_col]].copy()
    X_orig = np.array(df_selected[X_col], dtype=float)
    y_orig = np.array(df_selected[Y_col], dtype=float)
except KeyError as e:
    print(f"Error: One of the required columns {e} was not found in the dataset.")
    exit()

print(f"--- Analysis Start ---")
print(f"Initial number of samples used: {len(X_orig)}")

# --- Part 1: Original Dataset Analysis ---

def fit_polynomial_aic(x, y):
    """Fits polynomial models and selects the best degree using AIC."""
    n = len(x)
    max_degree = max(1, n // 4)
    best_degree = 1
    min_aic = float('inf')
    coeffs_list = []
    log_y_hat_list = []

    # Fit for degrees 1 up to min(3, max_degree)
    for degree in range(1, min(4, max_degree + 1)):
        # Must fit in log space as Y represents cycles (positive quantity)
        log_y = np.log(y)
        
        try:
            coeffs = np.polyfit(x, log_y, degree)
            log_y_hat = np.polyval(coeffs, x)
            rss = np.sum((log_y - log_y_hat) ** 2)
            # AIC formula for regression: N * log(RSS/N) + 2k (where k is number of parameters = degree + 1)
            aic = n * np.log(rss / n) + 2 * (degree + 1)
        except RuntimeWarning as e:
             # This can happen if variance is zero, etc., but we proceed with the calculation structure
             print(f"Warning during degree {degree} fitting: {e}")
             aic = float('inf')

        print(f"  degree={degree}  AIC={aic:.2f}")
        
        if aic < min_aic:
            min_aic = aic
            best_degree = degree
            coeffs_list.append(coeffs)
            log_y_hat_list.append(np.polyval(coeffs, x))

    return best_degree, coeffs_list[-1], log_y_hat_list[-1] # Return the last calculated set for simplicity if multiple degrees were tested

# 1. Fit on Original Data
best_degree_orig, coeffs_orig, log_y_hat_orig = fit_polynomial_aic(X_orig, y_orig)

print("\n--- Part 1 Results (Original Data) ---")
print(f"Selected best degree: {best_degree_orig}")
print("Fitted coefficients:", coeffs_orig)

# Calculate R^2 for the original fit on log scale residuals vs predicted log values
log_y_pred_orig = np.polyval(coeffs_orig, X_orig)
r2_orig = r2_score(np.log(y_orig), log_y_pred_orig)
print(f"R-squared (Log Scale): {r2_orig:.4f}")

# Compute Uncertainty Band for Original Data
x_range_orig = np.linspace(X_orig.min(), X_orig.max(), 200)
residuals_orig = np.log(y_orig) - log_y_hat_orig # Shape (n,) computed at original x
boot_preds_orig = np.zeros((500, 200))
rng_orig = np.random.default_rng(seed=42)

for i in range(500):
    resampled = rng_orig.choice(residuals_orig, size=len(X_orig), replace=True)
    log_y_boot = log_y_hat_orig + resampled # shape (n,) — add to base at original x
    c = np.polyfit(X_orig, log_y_boot, best_degree_orig)
    boot_preds_orig[i] = np.polyval(c, x_range_orig)   # shape (200,)

lower_orig = np.clip(np.exp(np.percentile(boot_preds_orig, 3, axis=0)), 0, None)
upper_orig = np.clip(np.exp(np.percentile(boot_preds_orig, 97, axis=0)), 0, y_orig.max() * 2)
mean_orig = np.exp(np.polyval(coeffs_orig, x_range_orig))

# Plotting Part 1
plt.figure(figsize=(10, 6))
plt.scatter(X_orig, y_orig, color="steelblue", zorder=5, label="Data")
plt.fill_between(x_range_orig, lower_orig, upper_orig, alpha=0.3, color="crimson", label="94% band (Original)")
plt.plot(x_range_orig, mean_orig, color="blue", linewidth=2, label=f"Fit (degree={best_degree_orig})")
plt.xlabel("CT Sv (um)")
plt.ylabel("Number of Cycles to Failure")
plt.title("Fatigue Life Prediction: Original Data Fit")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("./results/output_plot_part1.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Part 2: Augmented Dataset Analysis ---

# Augment Data
std_dev = 1.5 # Small standard deviation for Gaussian noise
synthetic_points = []
for i in range(10):
    # Generate synthetic points along the fitted curve (using mean_orig prediction)
    x_synth = x_range_orig[np.random.randint(0, 200)]
    y_synth = np.random.normal(mean_orig[np.random.randint(0, 200)], std_dev * (1 + (X_orig.max() - X_orig.min()) / 50), 1)[0] # Scale noise slightly with range
    synthetic_points.append([x_synth, y_synth])

# Add Gaussian Noise to Original Data
noise = np.random.normal(0, std_dev, size=(len(X_orig), 1))
y_noisy_orig = y_orig + noise[:, 0]

# Combine datasets
X_aug = np.hstack([X_orig, np.array([0]*len(X_orig))]).astype(float) # Keep X coordinates consistent for structure if needed, but we use original X values
Y_aug = np.concatenate((y_noisy_orig, np.array([p[1] for p in synthetic_points])))
X_aug_coords = np.concatenate((X_orig, np.array([p[0] for p in synthetic_points])))

# Create the augmented DataFrame structure (conceptually)
df_augmented = pd.DataFrame({
    'ct_sv': X_aug_coords,
    'cycles': Y_aug
})

print(f"\nAugmented number of samples used: {len(X_aug_coords)}")

# 2. Fit on Augmented Data
best_degree_aug, coeffs_aug, log_y_hat_aug = fit_polynomial_aic(X_aug_coords, Y_aug)

print("\n--- Part 2 Results (Augmented Data) ---")
print(f"Selected best degree: {best_degree_aug}")
print("Fitted coefficients:", coeffs_aug)

# Calculate R^2 for the augmented fit on log scale residuals vs predicted log values
log_y_pred_aug = np.polyval(coeffs_aug, X_aug_coords)
r2_aug = r2_score(np.log(Y_aug), log_y_pred_aug)
print(f"R-squared (Log Scale): {r2_aug:.4f}")

# Compute Uncertainty Band for Augmented Data
x_range_aug = np.linspace(X_aug_coords.min(), X_aug_coords.max(), 200)
residuals_aug = np.log(Y_aug) - log_y_hat_aug # Shape (n,) computed at original x points in the augmented set
boot_preds_aug = np.zeros((500, 200))
rng_aug = np.random.default_rng(seed=42)

for i in range(500):
    resampled = rng_aug.choice(residuals_aug, size=len(X_aug_coords), replace=True)
    log_y_boot = log_y_hat_aug + resampled # shape (n,) — add to base at original x
    c = np.polyfit(X_aug_coords, log_y_boot, best_degree_aug)
    boot_preds_aug[i] = np.polyval(c, x_range_aug)   # shape (200,)

lower_aug = np.clip(np.exp(np.percentile(boot_preds_aug, 3, axis=0)), 0, None)
upper_aug = np.clip(np.exp(np.percentile(boot_preds_aug, 97, axis=0)), 0, Y_aug.max() * 2)
mean_aug = np.exp(np.polyval(coeffs_aug, x_range_aug))

# Plotting Part 2
plt.figure(figsize=(10, 6))
# Original Data (Distinct Color/Marker)
plt.scatter(X_orig, y_orig, color="steelblue", zorder=5, label="Original Data")
# Synthetic Points (Distinct Color/Marker)
plt.scatter([p[0] for p in synthetic_points], [p[1] for p in synthetic_points], color="orange", marker='x', s=50, zorder=6, label="Synthetic Points")

# Uncertainty Band and Fit Curve
plt.fill_between(x_range_aug, lower_aug, upper_aug, alpha=0.3, color="crimson", label="94% band (Augmented)")
plt.plot(x_range_aug, mean_aug, color="blue", linewidth=2, label=f"Fit (degree={best_degree_aug})")

# Plotting elements must include legend and grid
plt.xlabel("CT Sv (um)")
plt.ylabel("Number of Cycles to Failure")
plt.title("Fatigue Life Prediction: Augmented Data Fit")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("./results/output_plot_part2.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n--- Analysis Complete ---")
print("Successfully performed two polynomial fits (Original vs Augmented Data), computed uncertainty bands using residual bootstrap, and generated two comparison plots saved as PNG files.")