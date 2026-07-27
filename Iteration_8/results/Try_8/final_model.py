import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Data Loading
df = pd.read_csv("data/tensile_data.csv")

# --- Feature Selection and Preprocessing ---

# Select relevant columns for visualization: Strength (Y), Void Content (X), Fiber Type (Grouping)
numeric_cols = ['density_g_cm3', 'layer_count', 'curing_temperature_c', 'fiber_volume_fraction', 'void_content_pct', 'tensile_strength_mpa']
categorical_cols = ['fiber_type', 'resin_type']

# Ensure the target variable and primary predictors are float numpy arrays
df['tensile_strength_mpa'] = df['tensile_strength_mpa'].astype(float)
df['void_content_pct'] = df['void_content_pct'].astype(float)
df['fiber_type'] = df['fiber_type'].astype('category')

# --- Visualization: Tensile Strength vs. Void Content, grouped by Fiber Type ---

plt.figure(figsize=(10, 7))

# Iterate through unique fiber types to create distinct scatter plots/groups
for fiber in df['fiber_type'].unique():
    subset = df[df['fiber_type'] == fiber]
    
    # Scatter plot for each group
    plt.scatter(
        subset['void_content_pct'], 
        subset['tensile_strength_mpa'], 
        label=str(fiber), 
        alpha=0.6, 
        s=20
    )

# Add general trend line (optional, but improves publication quality)
# Fit a simple linear regression model for the entire dataset to show overall trend
X = df['void_content_pct'].values.reshape(-1, 1)
Y = df['tensile_strength_mpa'].values
try:
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X, Y)
    y_pred = model.predict(X)
    plt.plot(X, y_pred, color='red', linestyle='--', label=f'Overall Trend (R²={model.score(X, Y):.2f})')
except ImportError:
    pass # Skip trend line if sklearn is unavailable or fails

# Setting labels and title based on schema/request context
plt.xlabel("Void Content (%)")
plt.ylabel("Tensile Strength (MPa)")
plt.title("Influence of Void Content on Tensile Strength Across Different Fiber Types")
plt.legend(title="Fiber Type")
plt.grid(True, linestyle=':', alpha=0.7)

# Save the figure
plt.savefig("./results/output_plot.png", dpi=150, bbox_inches="tight")

print("Successfully generated publication-quality scatter plot showing Tensile Strength vs. Void Content.")