# by Avaya Khatri
import numpy as np

# 3 inputs
X = np.array([[2.0, 1.0, 3.0]])
y = np.array([[4.0]])

lr = 0.05

# initial weights
W1 = np.array([[ 0.5, -0.1,  0.2],
               [ 0.1,  0.3, -0.4],
               [ 0.2, -0.2, 0.1]])
W2 = np.array([[1.0],
               [0.5],
               [-1.0]])

print(" INITIAL WEIGHTS ")
print("W1:\n", W1)
print("W2:\n", W2, "\n")

# activation functions
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# forwards pass
Z1 = np.dot(X, W1)
H1 = relu(Z1)
y_pred = np.dot(H1, W2)

print("  FORWARD PASS  ")
print("Z1 (Input to Hidden):", Z1)
print("H1 (Hidden after ReLU):", H1)
print("y_pred (Output):", y_pred, "\n")

# loss
dy_pred = y_pred - y
loss = 0.5 * np.sum(dy_pred ** 2)

print("  LOSS  ")
print("Raw Error (y_pred - y):", dy_pred)
print("MSE Loss:", loss, "\n")

# backpropagation
dW2 = np.dot(H1.T, dy_pred)
dH1 = np.dot(dy_pred, W2.T)
dZ1 = dH1 * relu_derivative(Z1)
dW1 = np.dot(X.T, dZ1)

print("  CALCULATED GRADIENTS  ")
print("dW1:\n", dW1)
print("dW2:\n", dW2, "\n")

# weights update
W1_new = W1 - lr * dW1
W2_new = W2 - lr * dW2

print("  UPDATED WEIGHTS ")
print("W1_new:\n", np.round(W1_new, 4))
print("W2_new:\n", np.round(W2_new, 4), "\n")

# new prediction
new_Z1 = np.dot(X, W1_new)
new_H1 = relu(new_Z1)
new_prediction = np.dot(new_H1, W2_new)

print(" FINAL RESULT ")
print("Old Prediction:", y_pred)
print("New Prediction:", new_prediction)
print("Prediction should be closer to target 4.0!")