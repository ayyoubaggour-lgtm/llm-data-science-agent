import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import os

# Setup directory for saving results
os.makedirs("./results", exist_ok=True)

# Data Loading
df = pd.read_csv("data/tensile_data.csv")

# Column Selection and Preparation
X_col = 'fiber_volume_fraction'
Y_col = 'tensile_strength_mpa'

# Convert selected numeric columns to float numpy arrays
X = np.array(df[X_col], dtype=float)
Y = np.array(df[Y_col], dtype=float)

# Perform linear regression (Y ~ X)
slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

# Calculate predicted values and residuals
Y_predicted = slope * X + intercept
residuals = Y - Y_predicted

# Create the residual plot: Residuals vs Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(Y_predicted, residuals, alpha=0.5)
plt.xlabel("Predicted Tensile Strength (MPa)")
plt.ylabel("Residuals (Observed - Predicted) (MPa)")
plt.title("Residual Plot for Tensile Strength vs Fiber Volume Fraction")

# Add a horizontal line at y=0 for reference
plt.axhline(y=0, color='r', linestyle='--')

# Save the plot
plt.savefig("./results/output_plot.png", dpi=150, bbox_inches="tight")
plt.close()

print("Successfully generated and saved the residual plot to ./results/output_plot.png.")