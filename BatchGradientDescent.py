import numpy as np
from sklearn.preprocessing import StandardScaler

class BatchGradientDescent:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.scaler = None  

    def fit(self, X, y):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Add a column of ones for the bias term
        X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

        # Initialize weights randomly
        self.theta = np.random.randn(X_b.shape[1], 1)

        y = y.values.reshape(-1, 1)

        # Batch Gradient Descent
        for i in range(self.iterations):
            predictions = X_b.dot(self.theta)
            gradients = (2 / X_b.shape[0]) * X_b.T.dot(predictions - y)
            self.theta -= self.learning_rate * gradients

            # Print loss after every 100 iterations
            if i % 100 == 0:
                loss = np.mean((predictions - y) ** 2)
                print(f"Iteration {i}, Loss: {loss}")

        return self.theta

    def predict(self, X):
        # Transform new input data using the same scaler
        X_scaled = self.scaler.transform(X)
        X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
        return X_b.dot(self.theta)
