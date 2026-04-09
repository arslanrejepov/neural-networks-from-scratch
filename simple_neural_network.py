import numpy as np

# -----------------------------
# Activation Functions
# -----------------------------
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return Z > 0

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# -----------------------------
# Loss Function
# -----------------------------
def compute_loss(y_true, y_pred):
    m = y_true.shape[0]
    loss = - (1/m) * np.sum(
        y_true * np.log(y_pred + 1e-8) +
        (1 - y_true) * np.log(1 - y_pred + 1e-8)
    )
    return loss

# -----------------------------
# Initialize Parameters
# -----------------------------
def initialize_parameters(input_size, hidden_size, output_size):
    np.random.seed(42)

    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))

    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))

    return W1, b1, W2, b2

# -----------------------------
# Forward Propagation
# -----------------------------
def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    cache = (Z1, A1, Z2, A2)
    return A2, cache

# -----------------------------
# Backpropagation
# -----------------------------
def backward(X, y, W2, cache):
    m = X.shape[0]
    Z1, A1, Z2, A2 = cache

    # Output layer gradient
    dZ2 = A2 - y
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

    # Hidden layer gradient
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1/m) * np.dot(X.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

# -----------------------------
# Update Parameters
# -----------------------------
def update(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    return W1, b1, W2, b2

# -----------------------------
# Training Loop
# -----------------------------
def train(X, y, hidden_size=4, lr=0.1, epochs=1000):
    input_size = X.shape[1]
    output_size = 1

    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        # Forward
        y_pred, cache = forward(X, W1, b1, W2, b2)

        # Loss
        loss = compute_loss(y, y_pred)

        # Backward
        dW1, db1, dW2, db2 = backward(X, y, W2, cache)

        # Update
        W1, b1, W2, b2 = update(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return W1, b1, W2, b2

# -----------------------------
# Prediction
# -----------------------------
def predict(X, W1, b1, W2, b2):
    y_pred, _ = forward(X, W1, b1, W2, b2)
    return (y_pred > 0.5).astype(int)

# -----------------------------
# Example Usage (XOR problem)
# -----------------------------
if __name__ == "__main__":
    # XOR dataset
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([[0], [1], [1], [0]])

    # Train
    W1, b1, W2, b2 = train(X, y, hidden_size=4, lr=0.1, epochs=2000)

    # Predict
    preds = predict(X, W1, b1, W2, b2)

    print("\nPredictions:")
    print(preds)