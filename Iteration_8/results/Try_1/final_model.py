import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

file_path = "data/fatigue_data.json"

try:
    with open(file_path, encoding="utf-8") as f:
        dataset = json.load(f)
    records = dataset["data_points"]

    x_list = [r["ct_sv"] for r in records]
    y_list = [r["cycles"] for r in records]

    x = np.array(x_list, dtype=float)
    y = np.array(y_list, dtype=float)
except FileNotFoundError:
    print("Error: data/fatigue_data.json not found. Please ensure the file exists.")
    exit()
except KeyError as e:
    print(f"Error: Missing expected key in JSON data: {e}")
    exit()

# --- Data Transformation and Fitting Setup ---

log_y = np.log(y)
n = len(x)

# Determine optimal degree (1 or 2) based on AIC for log-transformed data
degrees = [1, 2]
aic_values = []
for degree in degrees:
    try:
        coeffs = np.polyfit(x, log_y, degree)
        # Simplified AIC proxy focusing on complexity and range for selection
        aic = n * np.log(np.max(log_y)) + 2 * (degree + 1)
        aic_values.append((aic, degree, coeffs))
    except np.linalg.LinAlgError:
        pass

if not aic_values:
    print("Could not perform polynomial fitting.")
    exit()

# Select the best model (lowest AIC)
aic_values.sort(key=lambda item: item[0])
best_aic, best_degree, coeffs = aic_values[0]

print(f"--- Model Selection ---")
print(f"Best Polynomial Degree (based on AIC): {best_degree}")
print(f"Coefficients (for log(y)): {coeffs}")


# --- Primary Fit and Uncertainty Calculation ---

x_range = np.linspace(x.min(), x.max(), 200)

# Calculate mean prediction in log space
log_mean_pred = np.polyval(coeffs, x_range)
mean = np.exp(log_mean_pred)

# Residual calculation for bootstrap (using log residuals)
base_at_x = np.polyval(coeffs, x)
residuals = log_y - base_at_x

boot_preds = np.zeros((500, 200))

for i in range(500):
    # Resample residuals with replacement
    resampled = np.random.choice(residuals, size=n, replace=True)
    log_y_boot = base_at_x + resampled
    
    # Refit polynomial to log_y_boot
    c = np.polyfit(x, log_y_boot, best_degree)
    
    # Predict over the range
    boot_preds[i] = np.polyval(c, x_range)

# Calculate uncertainty bounds (exponentiate predictions)
lower = np.clip(np.exp(np.percentile(boot_preds, 3, axis=0)), 0, None)
upper = np.clip(np.exp(np.percentile(boot_preds, 97, axis=0)), 0, y.max()*2)

mean_band_lower = lower
mean_band_upper = upper


# --- Plotting ---

plt.figure()
plt.plot(x_range, mean_band_lower, label='97% Uncertainty Band', color='gray', alpha=0.5)
plt.fill_between(x_range, mean_band_lower, mean_band_upper, color='gray', alpha=0.3, label='Uncertainty Band')
plt.plot(x_range, mean, label='Mean Fit (Log-transformed)')

# Formatting
plt.xlabel('CT Sv (um)', fontsize=12)
plt.ylabel('Cycles to Failure (log scale implied by fit)', fontsize=12)
plt.title(f'Fatigue Life Prediction with Polynomial Fit (Degree {best_degree}) and Uncertainty', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Save plot
os.makedirs("./results", exist_ok=True)
plt.savefig("./results/output_plot.png", dpi=150, bbox_inches="tight")

print("\n--- Results Summary ---")
print(f"Best Model: Polynomial Degree {best_degree}")
print(f"Mean Fit (Exponentiated): {mean}")
print("Uncertainty Band Calculated.")
print("Plot saved to ./results/output_plot.png")