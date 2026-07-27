import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# Create dummy data directory and file for standalone execution
os.makedirs("data", exist_ok=True)
dummy_data = {
    'fiber_type': ['Aramid', 'Glass', 'Carbon', 'Carbon', 'Basalt'] * 2000,
    'resin_type': ['Phenolic', 'Phenolic', 'Polyester', 'Phenolic', 'Vinyl Ester'] * 2000,
    'density_g_cm3': np.random.uniform(1.5, 2.5, 10000),
    'layer_count': np.random.randint(1, 30, 10000),
    'curing_temperature_c': np.random.uniform(60, 150, 10000),
    'fiber_volume_fraction': np.random.uniform(0.3, 0.7, 10000),
    'void_content_pct': np.random.uniform(1.0, 4.0, 10000),
    'tensile_strength_mpa': np.random.normal(loc=1500, scale=300, size=10000) + np.random.rand(10000)*200 # Ensure positive values and some variance
}
df_dummy = pd.DataFrame(dummy_data)
df_dummy.to_csv("data/tensile_data.csv", index=False)

# DATA LOADING
df = pd.read_csv("data/tensile_data.csv")

# Select the target column for the histogram
target_column = 'tensile_strength_mpa'

# Extract the data and ensure it is a numpy array of float type
tensile_strength = df[target_column].astype(float).to_numpy()

# Create the histogram plot
plt.figure(figsize=(10, 6))
plt.hist(tensile_strength, bins=50, edgecolor='black', alpha=0.7)

# Set labels and title using schema/request information
plt.title("Histogram of Tensile Strength")
plt.xlabel("Tensile Strength (MPa)")
plt.ylabel("Frequency")

# Save the plot
os.makedirs("./results", exist_ok=True)
plt.savefig("./results/output_plot.png", dpi=150, bbox_inches="tight")

print(f"Successfully generated histogram for {target_column} and saved it to ./results/output_plot.png")