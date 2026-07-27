import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# Data Loading
df = pd.read_csv("data/tensile_data.csv")

# --- Feature Engineering for Clustering ---
# Identify categorical and numeric columns
categorical_cols = ['fiber_type', 'resin_type']
numeric_cols = [col for col in df.columns if col not in categorical_cols]

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Select the feature matrix X (all encoded and numeric columns)
X = df_encoded.copy()

# Scale the data for k-means clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- K-Means Clustering ---
# Determine optimal number of clusters (K). Since none is specified, we choose K=4 as a reasonable default.
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels back to the original DataFrame for inspection/summary
df['Cluster'] = cluster_labels

# --- Output and Visualization ---

print("Clustering completed using k-means on all available features.")
print(f"The data has been grouped into {k} clusters. Cluster distribution:\n{df['Cluster'].value_counts().sort_index()}")

# Since the request is general clustering, we will visualize the cluster assignment relative to a key variable 
# (e.g., tensile strength) for visualization purposes, although this isn't strictly required by k-means output.
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['fiber_volume_fraction'], df['tensile_strength_mpa'], c=cluster_labels, cmap='viridis', alpha=0.7)

plt.xlabel('fiber_volume_fraction')
plt.ylabel('tensile_strength_mpa')
plt.title('K-Means Clustering of Tensile Samples (k=4)')
plt.colorbar(scatter, ticks=range(k), label='Cluster ID')

# Save the plot
os.makedirs("./results", exist_ok=True)
plt.savefig("./results/output_plot.png", dpi=150, bbox_inches="tight")
plt.close()