from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from BatchGradientDescent import BatchGradientDescent  
from StochasticGradientDescent import StochasticGradientDescent  

# Function to predict house value for MedInc = 8.0 using Batch Gradient Descent
def predict_med_income_80000_batch():
    feature_names = df.drop("MedHouseVal", axis=1).columns

    new_district = np.array([[8.0] + list(df[feature_names].mean()[1:].values)])
    new_district_df = pd.DataFrame(new_district, columns=feature_names)

    new_district_scaled = bgd.scaler.transform(new_district_df)
    new_district_scaled = np.c_[np.ones((new_district_scaled.shape[0], 1)), new_district_scaled]

    predicted_price = new_district_scaled.dot(theta_bgd)
    print("Predicted House Value for MedInc = 8.0 (Batch):", predicted_price[0, 0])

# Function to predict house value for MedInc = 8.0 using Stochastic Gradient Descent
def predict_med_income_80000_stochastic():
    feature_names = df.drop("MedHouseVal", axis=1).columns

    new_district = np.array([[8.0] + list(df[feature_names].mean()[1:].values)])
    new_district_df = pd.DataFrame(new_district, columns=feature_names)

    new_district_scaled = sgd.scaler.transform(new_district_df)
    new_district_scaled = np.c_[np.ones((new_district_scaled.shape[0], 1)), new_district_scaled]

    predicted_price = new_district_scaled.dot(theta_sgd)
    print("Predicted House Value for MedInc = 8.0 (SGD):", predicted_price[0, 0])


california_data = fetch_california_housing()
df = pd.DataFrame(california_data.data, columns=california_data.feature_names)
df['MedHouseVal'] = california_data.target  


print(df.head())
print("\nBasic Statistics:")
print(df.describe())

# Scatter plot for MedInc vs. MedHouseVal
plt.figure(figsize=(10, 6))
plt.scatter(df['MedInc'], df['MedHouseVal'], alpha=0.5)
plt.xlabel('Median Household Income ($10,000s)')
plt.ylabel('Median House Value ($100,000s)')
plt.title('Scatter Plot of MedInc vs. MedHouseVal')

# Correlation between MedInc and MedHouseVal
correlation = df['MedInc'].corr(df['MedHouseVal'])
print(f"Correlation between MedInc and MedHouseVal: {correlation}")

# Median values of MedInc and MedHouseVal
median_medinc = df['MedInc'].median()
median_medhouseval = df['MedHouseVal'].median()
print(f"Median value of MedInc (Median Household Income): {median_medinc}")
print(f"Median value of MedHouseVal (Median House Value): {median_medhouseval}")

# Split the data into features (X) and target (y)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Batch Gradient Descent 
print("\nBATCH GRADIENT DESCENT:")
bgd = BatchGradientDescent(learning_rate=0.01, iterations=1000)
theta_bgd = bgd.fit(X_train, y_train)  # Train model
print("Trained model parameters (theta) - Batch Gradient Descent:", theta_bgd)

# Predict on test set
predictions_bgd = bgd.predict(X_test)

# Scatter plot of actual vs predicted values (BatchGD)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions_bgd, alpha=0.5)
plt.xlabel('Actual House Values')
plt.ylabel('Predicted House Values')
plt.title('Actual vs Predicted House Values - Batch Gradient Descent')

# Predict for MedInc = 8.0 (Batch graduient Descent)
predict_med_income_80000_batch()

# Stochastic Gradient Descent ------
print("\nSTOCHASTIC GRADIENT DESCENT:")
sgd = StochasticGradientDescent(learning_rate=0.01, iterations=1000)
theta_sgd = sgd.fit(X_train, y_train)  # Train model
print("Trained model parameters (theta) - Stochastic Gradient Descent:", theta_sgd)

# Predict on test set
predictions_sgd = sgd.predict(X_test)

# Scatter plot of actual vs predicted values (SGD)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions_sgd, alpha=0.5)
plt.xlabel('Actual House Values')
plt.ylabel('Predicted House Values')
plt.title('Actual vs Predicted House Values - Stochastic Gradient Descent')

# Predict for MedInc = 8.0 (SGD)
predict_med_income_80000_stochastic()

# Regression Line (Batch) ----
plt.figure(figsize=(10, 6))
sorted_indices = X_test['MedInc'].argsort()  
X_test_sorted = X_test.iloc[sorted_indices]
y_test_sorted = y_test.iloc[sorted_indices]
predictions_bgd_sorted = predictions_bgd[sorted_indices]

# Scatter plot of actual test data
plt.scatter(X_test['MedInc'], y_test, alpha=0.5, label="Actual Data")
plt.plot(X_test_sorted['MedInc'], predictions_bgd_sorted, color='red', label="BGD Regression Line", linewidth=2)

plt.xlabel('Median Household Income ($10,000s)')
plt.ylabel('Median House Value ($100,000s)')
plt.title('Regression Line - Batch Gradient Descent')
plt.legend()

# Regression Line (Stochastic gd) ------------
plt.figure(figsize=(10, 6))
predictions_sgd_sorted = predictions_sgd[sorted_indices]  

# Scatter plot of actual test data
plt.scatter(X_test['MedInc'], y_test, alpha=0.5, label="Actual Data")
plt.plot(X_test_sorted['MedInc'], predictions_sgd_sorted, color='green', label="SGD Regression Line", linewidth=2)

plt.xlabel('Median Household Income ($10,000s)')
plt.ylabel('Median House Value ($100,000s)')
plt.title('Regression Line - Stochastic Gradient Descent')
plt.legend()

plt.show()  
