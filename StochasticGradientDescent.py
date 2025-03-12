import numpy as np
from sklearn.preprocessing import StandardScaler

class StochasticGradientDescent:
    def __init__(self, learning_rate=0.01, iterations=100):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.scaler = None

    def fit(self, X, y):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Add intercept column
        X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

        # Initialize random weights
        self.theta = np.random.randn(X_b.shape[1], 1)

        y = y.values.reshape(-1, 1)

        m = X_b.shape[0]

        for iteration in range(self.iterations):
            indices = np.arange(m)
            # Shuffle data each epoch
            np.random.shuffle(indices)  

            for i in indices:
                x_i = X_b[i:i+1]
                y_i = y[i:i+1]

                prediction = x_i.dot(self.theta)
                gradient = 2 * x_i.T.dot(prediction - y_i)

                # Gradient clipping to avoid explosion
                np.clip(gradient, -10, 10, out=gradient)

                self.theta -= self.learning_rate * gradient

            # tracking loss every few epochs
            if iteration % 10 == 0:
                predictions = X_b.dot(self.theta)
                loss = np.mean((predictions - y) ** 2)
                print(f"Iteration {iteration}, Loss: {loss}")

        return self.theta

    def predict(self, X):
        X_scaled = self.scaler.transform(X)  
        X_b = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]
        return X_b.dot(self.theta)
