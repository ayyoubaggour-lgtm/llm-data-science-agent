import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# DATA LOADING
df = pd.read_csv("data/Iris.csv")

# Column Selection and Preprocessing
numeric_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
categorical_col = 'Species'

X_raw = df[numeric_cols]
y_raw = df[categorical_col]

# Handle Missing Values (Unconditional Check)
for col in X_raw.columns:
    if X_raw[col].isna().any():
        median_val = X_raw[col].median()
        X_raw[col] = X_raw[col].fillna(median_val)

# Convert to numpy arrays with float dtype for computation
X = np.array(X_raw.values, dtype=float)

# Encode Target Variable (Species)
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
class_names = le.classes_

print(f"Starting analysis with {df.shape[0]} samples.")

# Scaling Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print("Data split into training and testing sets.")

# Model Comparison Setup
models = {
    "Logistic Regression": LogisticRegression(random_state=42, solver='lbfgs'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(random_state=42, probability=True),
    "Gaussian Naive Bayes": GaussianNB()
}

results = {}

# Model Training and Evaluation Loop
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        conf_mat = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "conf_mat": conf_mat
        }
    except ValueError as e:
        print(f"Could not evaluate {name}: {e}")
        results[name] = None

# Reporting Results
print("\n=========================================================")
print("Classification Model Comparison (Iris Species Prediction)")
print("=========================================================")

header_format = f"{'Model':<25}{'Accuracy':<10}{'Precision':<12}{'Recall':<10}{'F1-Score':<12}"
print(header_format.format("", "", "", "", ""))

for name, res in results.items():
    if res:
        row = f"{name:<25}{res['accuracy']:<10.4f}{res['precision']:<12.4f}{res['recall']:<10.4f}{res['f1']:<12.4f}"
        print(row)

# Detailed Confusion Matrix Reporting
print("\n--- Detailed Metrics ---")
for name, res in results.items():
    if res:
        print(f"\nModel: {name}")
        print("Confusion Matrix:")
        print("Labels (Classes):", class_names)
        print("Rows are True Labels, Columns are Predicted Labels.")
        # Removed np.set_printoptions as it caused the error
        print(res['conf_mat'])

print("\nAnalysis complete. Results summarize the performance of various classifiers on the Iris dataset.")