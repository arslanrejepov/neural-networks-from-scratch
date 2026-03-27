from single_neuron import sigmoid
import numpy as np
def forward_propagation(x,w,b):
    z= np.dot(w,x)+b
    a = sigmoid(z)
    return a

x = np.array([1.0])
w=np.array([0.5])
b = 0.1

output = forward_propagation(x, w, b)
