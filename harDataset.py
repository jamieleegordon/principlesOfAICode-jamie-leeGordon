import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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

# Ensure unique feature names
if features_df["feature"].duplicated().any():
    features_df["feature"] = features_df["feature"] + "_" + features_df.index.astype(str)

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

# Ensure datasets loaded correctly
if X_train is None or y_train is None or X_test is None or y_test is None:
    raise RuntimeError("Error loading dataset files. Check file paths and formats.")

# Map the activity IDs to their names
y_train["Activity"] = y_train["Activity"].map(activity_map)
y_test["Activity"] = y_test["Activity"].map(activity_map)

# --- 2. CONVERT MULTI-CLASS TO BINARY ---
def to_binary_label(activity):
    if activity in ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]:
        return 1 # Active
    else:
        return 0 # Inactive

y_train["Binary"] = y_train["Activity"].apply(to_binary_label)
y_test["Binary"] = y_test["Activity"].apply(to_binary_label)

# Print dataset summary
print(f"Train set shape: {X_train.shape}, Labels shape: {y_train.shape}")
print(f"Test set shape: {X_test.shape}, Labels shape: {y_test.shape}")

print("Data successfully loaded and processed.")

def linear_kernel():
    # --- 3. BUILDING THE PIPELINE ---
    pipeline = Pipeline([
        ('scaler', StandardScaler()),       # Standardize the data
        ('pca', PCA(n_components=50)),      # Reduce from 561 features -> 50
        ('svc', SVC(kernel='linear'))       # Apply Support Vector Classification
    ])

    # --- 4. FITTING THE PIPELINE ---
    pipeline.fit(X_train, y_train["Binary"])

    # --- 5. PREDICTIONS ---
    y_pred = pipeline.predict(X_test)

    # --- 6. EVALUATING THE MODEL ---
    accuracy = accuracy_score(y_test["Binary"], y_pred)
    print(f"Linear Kernel Model accuracy: {accuracy * 100:.2f}%")

def polynomial_kernel():
    # --- 3. BUILDING THE PIPELINE ---
    pipeline = Pipeline([
        ('scaler', StandardScaler()),       
        ('pca', PCA(n_components=50)),      
        ('svc', SVC(kernel='poly', degree=3))                      
    ])

    # --- 4. FITTING THE PIPELINE ---
    pipeline.fit(X_train, y_train["Binary"])

    # --- 5. PREDICTIONS ---
    y_pred = pipeline.predict(X_test)

    # --- 6. EVALUATING THE MODEL ---
    accuracy = accuracy_score(y_test["Binary"], y_pred)
    print(f"Polynomial_kernel Kernel Model accuracy: {accuracy * 100:.2f}%")

def rbf_kernel():
    # --- 3. BUILDING THE PIPELINE ---
    pipeline = Pipeline([
        ('scaler', StandardScaler()),       
        ('pca', PCA(n_components=50)),      
        ('svc', SVC(kernel='rbf'))                      
    ])

    # --- 4. FITTING THE PIPELINE ---
    pipeline.fit(X_train, y_train["Binary"])

    # --- 5. PREDICTIONS ---
    y_pred = pipeline.predict(X_test)

    # --- 6. EVALUATING THE MODEL ---
    accuracy = accuracy_score(y_test["Binary"], y_pred)
    print(f"RBF Kernel Model accuracy: {accuracy * 100:.2f}%")

linear_kernel()
polynomial_kernel()
rbf_kernel()

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=50)),  
    ('svc', SVC())
])

# Parameter grid to search over
param_grid = [
    # Linear kernel
    {
        'svc__kernel': ['linear'],
        'svc__C': [0.01, 0.1, 1, 10, 100, 1000]
    },
    # Polynomial kernel
    {
        'svc__kernel': ['poly'],
        'svc__C': [0.1, 1, 10],
        'svc__degree': [2, 3, 4],
        'svc__gamma': [0.0001, 0.001, 0.01, 0.1, 1],
        'svc__coef0': [0.0, 0.1, 0.5]  # Controls influence of higher-degree terms
    },
    # RBF kernel
    {
        'svc__kernel': ['rbf'],
        'svc__C': [0.01, 0.1, 1, 10, 100],
        'svc__gamma': [0.0001, 0.001, 0.01, 0.1, 1]
    }
]


# GridSearchCV with pipeline and parameter grid
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring='accuracy',  # Or another multi-class metric, e.g., 'f1_micro'
    cv=3,  # 3-fold cross-validation (or 5-fold if feasible)
    n_jobs=-1,
    verbose=1
)

# Fit the grid search model
grid_search.fit(X_train, y_train["Binary"].values.ravel())


# Print the best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)





