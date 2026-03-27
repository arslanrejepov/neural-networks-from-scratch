import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s*(1-s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1-np.tanh(x)**2

def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return (x>0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x>0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x>0, 1, alpha)

