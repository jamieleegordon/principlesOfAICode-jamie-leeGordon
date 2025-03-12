from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Example pipeline (with optional PCA)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    # ('pca', PCA(n_components=50)),  # Uncomment if doing PCA
    ('svc', SVC())
])

# Parameter grid to search over
param_grid = [
    {
        'svc__kernel': ['linear'],
        'svc__C': [0.1, 1, 10, 100]
    },
    {
        'svc__kernel': ['poly'],
        'svc__C': [0.1, 1],
        'svc__degree': [2, 3],
        'svc__gamma': [0.001, 0.01, 0.1]
    },
    {
        'svc__kernel': ['rbf'],
        'svc__C': [0.1, 1, 10],
        'svc__gamma': [0.001, 0.01, 0.1]
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
grid_search.fit(X_train, y_train.values.ravel())

# Print the best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)
