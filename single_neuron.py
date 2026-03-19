import numpy as np

def weighted_sum(inputs, weights, bias):
    inputs  = np.array(inputs,  dtype=float)
    weights = np.array(weights, dtype=float)

    z = np.dot(inputs, weights) + bias
    return z

def step_activation(z):
    return 1 if z >= 0 else 0

def neuron(inputs, weights, bias, activation_fn=step_activation):
    z = weighted_sum(inputs, weights, bias)
    output = activation_fn
    return output