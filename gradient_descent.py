import numpy as np 

def update_parameters(w, b,dw,learning_rate):
    w = w -learning_rate * dw
    b = b-learning_rate * dw

    return w, b

def train(X, y, W, b, learning_rate, epochs):
    for epoch in range(epochs): 
        z= np.dot(W, X) + b 
        a = 1/ (1+np.exp(-z))

        loss = -np.mean(y * np.log(a) + (1 - y)  * np.log(1 - a))

        dz = a - y
        dW = np.dot(dz, X.T)/X.shape[1]
        db = np.mean(dz)

        W, b = update_parameters(W, b, dW, db, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

        return W, b
        