import numpy as np

# 3 inputs
X = np.array([[2.0, 1.0, 3.0]]) 
y = np.array([[4.0]])

lr = 0.05

W1 = np.array([[ 0.5, -0.1,  0.2], [ 0.1,  0.3, -0.4], [ 0.2, -0.2,  0.1]])

W2 = np.array([[ 1.0], [ 0.5], [-1.0]])

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

Z1 = np.dot(X, W1)
H1 = relu(Z1)
y_pred = np.dot(H1, W2)

dy_pred = y_pred - y
loss = 0.5 * np.sum(dy_pred ** 2)

dW2 = np.dot(H1.T, dy_pred)

dH1 = np.dot(dy_pred, W2.T)

dZ1 = dH1 * relu_derivative(Z1)

dW1 = np.dot(X.T, dZ1)

W1_new = W1 - lr * dW1
W2_new = W2 - lr * dW2

new_Z1 = np.dot(X, W1_new)
new_H1 = relu(new_Z1)
new_prediction = np.dot(new_H1, W2_new)

print("Old Prediction:", y_pred)
print("New Prediction:", new_prediction)