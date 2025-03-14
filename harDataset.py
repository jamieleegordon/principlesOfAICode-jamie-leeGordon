import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

# --- 1. LOADING THE DATA ---
PATH = "UCI HAR Dataset/"
features_path = PATH + "features.txt"
activity_labels_path = PATH + "activity_labels.txt"
X_train_path = PATH + "train/X_train.txt"
y_train_path = PATH + "train/y_train.txt"
X_test_path = PATH + "test/X_test.txt"
y_test_path = PATH + "test/y_test.txt"

# Load feature names
features_df = pd.read_csv(features_path, sep="\\s+", header=None, names=["idx", "feature"])
features_df["feature"] = features_df["feature"] + "_" + features_df.index.astype(str)  # Ensure unique names
feature_names = features_df["feature"].tolist()

# Load activity labels
activity_labels_df = pd.read_csv(activity_labels_path, sep="\\s+", header=None, names=["id", "activity"])
activity_map = dict(zip(activity_labels_df.id, activity_labels_df.activity))

# Function to load datasets safely
def load_data(file_path, column_names=None):
    try:
        return pd.read_csv(file_path, sep="\\s+", header=None, names=column_names)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Load train/test sets
X_train = load_data(X_train_path, feature_names)
y_train = load_data(y_train_path, ["Activity"])
X_test = load_data(X_test_path, feature_names)
y_test = load_data(y_test_path, ["Activity"])

if X_train is None or y_train is None or X_test is None or y_test is None:
    raise RuntimeError("Error loading dataset files. Check file paths and formats.")

# Map the activity IDs to their names
y_train["Activity"] = y_train["Activity"].map(activity_map)
y_test["Activity"] = y_test["Activity"].map(activity_map)

# --- 2. CONVERT MULTI-CLASS TO BINARY ---
def to_binary_label(activity):
    return 1 if activity in ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"] else 0  # 1 = Active, 0 = Inactive

y_train["Binary"] = y_train["Activity"].apply(to_binary_label)
y_test["Binary"] = y_test["Activity"].apply(to_binary_label)

print(f"Train set shape: {X_train.shape}, Labels shape: {y_train.shape}")
print(f"Test set shape: {X_test.shape}, Labels shape: {y_test.shape}")

print("Data successfully loaded and processed.")

# TRAIN & EVALUATE MODELS 
def evaluate_kernel(kernel_name, model, axes):
    print(f"\nEvaluating {kernel_name} Kernel...")

    # Fit model
    model.fit(X_train, y_train["Binary"])

    # Predict on test data
    y_pred = model.predict(X_test)

    # Compute accuracy
    accuracy = accuracy_score(y_test["Binary"], y_pred)
    print(f"{kernel_name} Kernel Accuracy: {accuracy * 100:.2f}%")

    # Compute confusion matrix
    cm = confusion_matrix(y_test["Binary"], y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Inactive", "Active"])

    # Plot confusion matrix
    disp.plot(cmap="Blues", ax=axes)
    axes.set_title(f"Confusion Matrix - {kernel_name} Kernel")

# Define parameter grid for polynomial kernel
param_grid_poly = {
    'svc__kernel': ['poly'],
    'svc__C': [0.1, 1, 10],
    'svc__degree': [2, 3, 4],
    'svc__gamma': [0.0001, 0.001, 0.01, 0.1, 1],
    'svc__coef0': [0.0, 0.1, 0.5]
}

# pipeline for grid search
poly_model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50)),
    ('svc', SVC())
])

# Perform grid search
grid_search = GridSearchCV(poly_model, param_grid=param_grid_poly, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train["Binary"])

# Print best parameters and accuracy
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_ * 100:.2f}%")

# Train the final model with the best parameters
best_poly_model = grid_search.best_estimator_

# --- RBF and Linear Models ---
linear_model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50)),
    ('svc', SVC(kernel='linear'))
])

rbf_model = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50)),
    ('svc', SVC(kernel='rbf'))
])

# display all confusion matrices at once
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# TRAIN & EVALUATE EACH MODEL 
evaluate_kernel("Linear", linear_model, axes[0])
evaluate_kernel("Polynomial", best_poly_model, axes[1])
evaluate_kernel("RBF", rbf_model, axes[2])

# Display the plot
plt.tight_layout()
plt.show()
