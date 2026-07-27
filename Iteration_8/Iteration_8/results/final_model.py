import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/Iris.csv")

# --- Data Preprocessing and Feature Selection ---

# Identify columns to use for unsupervised learning (PCA)
numeric_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
categorical_col = 'Species'
id_col = 'Id'

# Check for and drop ID column if it exists and is purely an index
if id_col in df.columns:
    df = df.drop(columns=[id_col])
    print(f"Dropped identifier column: {id_col}")

# Separate features (X) from labels/categories (y_labels)
# For PCA, we use all numeric measurements and exclude the label 'Species'
feature_cols = [col for col in df.columns if col not in ['Species']]
label_col = 'Species'

print(f"Original dataset shape: {df.shape}")

# Handle Missing Values (NaNs) - Check all relevant columns
missing_check_cols = feature_cols + [label_col]
for col in missing_check_cols:
    if df[col].isna().any():
        print(f"Warning: Column '{col}' contains NaN values. Imputing with median.")

# Check for NaN in feature matrix (X)
X_df = df[feature_cols].copy()
nan_count_before = X_df.isnull().sum().sum()
if nan_count_before > 0:
    print(f"Found {nan_count_before} missing values in features. Dropping rows with any NaN.")
    X_df.dropna(inplace=True)

# Check for NaN in label column (y_labels) - although we won't use it for PCA fitting, good practice to check
y_labels_series = df[label_col].copy()
nan_count_labels = y_labels_series.isna().sum()
if nan_count_labels > 0:
    print(f"Warning: Label column '{label_col}' contains {nan_count_labels} NaN values.")

# Re-align the label series to match the rows remaining after dropping NaNs in X
df_cleaned = df.loc[X_df.index]
y_labels_final = df_cleaned[label_col]


print(f"Shape after NaN handling: {X_df.shape}")

# Convert features to numpy array of floats
X = X_df.values.astype(float)

# --- PCA Implementation ---

# 1. Scaling the data (Crucial for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Applying PCA to reduce dimensions to 2
n_components = 2
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# --- Reporting Results ---

# Explained Variance Ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("\n--- PCA Analysis Summary ---")
print("Explained variance ratio for the first two components:")
print(f"Component 1: {explained_variance_ratio[0]:.4f}")
print(f"Component 2: {explained_variance_ratio[1]:.4f}")
print(f"Total explained variance by 2 components: {np.sum(explained_variance_ratio):.4f}")

# Feature names used for PCA (for reporting)
pca_feature_names = ", ".join(feature_cols)
print(f"\nFeatures used for PCA: {pca_feature_names} ({label_col} excluded — reserved for post-hoc comparison)")

# Store variables needed for visualization/next steps
X_reduced = X_pca
y_labels_for_plot = y_labels_final.values # Keep labels aligned with reduced data points

print("\nAnalysis complete. Variables 'X_reduced' (the 2D PCA coordinates) and 'y_labels_for_plot' (the corresponding species labels) are available for visualization.")

# --- Plotting (pass 2) ---

plt.figure(figsize=(10, 8))

# Get unique species labels and map them to integers for consistent plotting/coloring
categories = sorted(y_labels_final.unique())
print(f"Categories found in '{label_col}': {categories}")

# FIX: Use plt.get_cmap instead of plt.cm.get_cmap
palette = plt.get_cmap('viridis', len(categories))

# Scatter plot the reduced data points, colored by their original species label
for i, category in enumerate(categories):
    indices = y_labels_final[y_labels_final == category].index
    x_coords = X_reduced[indices, 0]
    y_coords = X_reduced[indices, 1]
    plt.scatter(x_coords, y_coords, color=palette(i), label=category, alpha=0.7)

# Add plot labels and title using schema information
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")
plt.title("PCA Projection of Iris Dataset onto First Two Components")

# Add a legend mapping colors to species names
plt.legend(title=label_col)

# Force axis limits based on the actual data range in X_reduced
x_min, x_max = np.min(X_reduced[:, 0]), np.max(X_reduced[:, 0])
y_min, y_max = np.min(X_reduced[:, 1]), np.max(X_reduced[:, 1])

# Calculate margins with a small buffer (e.g., 5% of the range)
x_margin = 0.05 * (np.ptp(X_reduced[:, 0])) if np.ptp(X_reduced[:, 0]) > 0 else 1.0
y_margin = 0.05 * (np.ptp(X_reduced[:, 1])) if np.ptp(X_reduced[:, 1]) > 0 else 1.0

plt.xlim(x_min - x_margin, x_max + x_margin)
plt.ylim(y_min - y_margin, y_max + y_margin)

# Save the plot
plt.savefig("./results/output_plot.png", dpi=150, bbox_inches="tight")