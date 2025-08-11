import numpy as np

class LogisticRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate   # α (learning rate)
        self.n_iters = n_iterations
        self.w = None             # Weights
        self.b = None             # Bias

    def sigmoid(self, z):
        """Sigmoid function: maps any real number to (0,1)"""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """Train the model using gradient descent"""
        m, n = X.shape              # m = number of samples, n = number of features
        self.w = np.zeros(n)        # initialize weights to zeros
        self.b = 0                  # initialize bias to zero

        for _ in range(self.n_iters):
            # Linear prediction: z = Xw + b
            z = np.dot(X, self.w) + self.b
            # Apply sigmoid to get predicted probabilities
            y_hat = self.sigmoid(z)

            # Compute gradients
            dw = (1/m) * np.dot(X.T, (y_hat - y))      # ∂J/∂w
            db = (1/m) * np.sum(y_hat - y)             # ∂J/∂b

            # Update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_prob(self, X):
        """Return predicted probabilities"""
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)

    def predict(self, X):
        """Return binary predictions (0 or 1)"""
        y_prob = self.predict_prob(X)
        return np.where(y_prob >= 0.5, 1, 0)


# ===== Example Usage =====

# Example dataset (AND gate)
X_train = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_train = np.array([0, 0, 0, 1])

# Create and train model
model = LogisticRegressionGD(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

# Predictions
print("Predicted probabilities:", model.predict_prob(X_train))
print("Predicted labels:", model.predict(X_train))
