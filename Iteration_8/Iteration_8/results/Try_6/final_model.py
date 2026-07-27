import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# Data Loading
df = pd.read_csv("data/tensile_data.csv")

# Identify numeric columns for correlation analysis
numeric_cols = [
    'density_g_cm3', 
    'layer_count', 
    'curing_temperature_c', 
    'fiber_volume_fraction', 
    'void_content_pct', 
    'tensile_strength_mpa'
]

# Select and convert relevant columns to float numpy arrays
df_numeric = df[numeric_cols].copy()
X = df_numeric.apply(lambda col: np.array(col, dtype=float))

# Perform Pearson Correlation Analysis (Correlation Matrix)
correlation_matrix = X.corr(method='pearson')

print("--- Pearson Correlation Analysis Summary ---")
print("The correlation matrix shows the linear relationship between all numeric features:")
print(correlation_matrix)

# Visualization: Plotting the relationship between Tensile Strength and Void Content
plt.figure(figsize=(10, 6))
plt.scatter(df['void_content_pct'], df['tensile_strength_mpa'], alpha=0.5)
plt.xlabel('void_content_pct')
plt.ylabel('tensile_strength_mpa')
plt.title('Pearson Correlation: Tensile Strength vs. Void Content')

# Save the plot
os.makedirs("./results", exist_ok=True)
plt.savefig("./results/output_plot.png", dpi=150, bbox_inches="tight")

print("\nSuccessfully performed Pearson correlation analysis and saved the visualization to ./results/output_plot.png.")