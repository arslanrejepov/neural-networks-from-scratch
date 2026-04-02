import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivative_sigmoid(x):
    s = sigmoid(x)
    return s * (1-s)

def relu(x):
    return np.maximum(0,x)

def derivative_relu(x):
    return (x > 0).astype(float)

def backward(X, Y, W1, b1, W2, b2):
    m =np.shape[1]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

        # ===== Backward =====

    # Output layer
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    # Hidden layer
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * derivative_relu(Z1)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return dW1, db1, dW2, db2