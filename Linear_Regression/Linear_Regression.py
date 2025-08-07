import numpy as np
import matplotlib.pyplot as plt

# 1. Generate some simple data
np.random.seed(0)
X = np.linspace(0, 10, 50)  # 50 data points between 0 and 10
y = 2.5 * X + np.random.randn(50) * 2  # True relation: y = 2.5x + noise

# 2. Initialize parameters
w = 0.0  # weight
b = 0.0  # bias
alpha = 0.01  # learning rate
epochs = 1000  # number of training steps
n = len(X)

# 3. Gradient Descent Loop
for epoch in range(epochs):
    # Predictions
    y_pred = w * X + b
    
    # Compute errors
    error = y_pred - y
    
    # Compute gradients
    dw = (2/n) * np.dot(error, X)  # ∂J/∂w
    db = (2/n) * np.sum(error)     # ∂J/∂b

    # Update parameters
    w -= alpha * dw
    b -= alpha * db

    # Optional: print loss every 100 steps
    if epoch % 100 == 0:
        loss = np.mean(error**2)
        print(f"Epoch {epoch}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")

# 4. Final results
print(f"\nFinal Model: y = {w:.2f}x + {b:.2f}")

# 5. Plot
plt.scatter(X, y, label='Data Points')
plt.plot(X, w * X + b, color='red', label='Fitted Line')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression Fit using Gradient Descent")
plt.show()
