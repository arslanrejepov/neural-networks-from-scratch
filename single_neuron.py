import numpy as np

x = np.array([1.0]) 
y_true = np.array([1.0])

w=np.random.randn()
b=np.random.randn()

#activation set sigmoid
def sigmoid(z):
    return 1/(1+np.exp(-z))

def forward(x,w,b):
    z=w*x+b
    pred=sigmoid(z)
    return pred

def loss(y_pred,y_true):
    return (y_pred-y_true)**2

lr=0.01

for epoch in range(100):
    
    y_pred = forward(x,w,b)
    l = loss(y_pred,y_true)

    dL_dy = 2*(y_pred-y_true)
    dy_dz = y_pred * (1-y_pred)

    dL_dw = dL_dy *dy_dz * x

    dL_db = dL_dy * dy_dz * 1

    w -= lr * dL_dw
    b -= lr * dL_db

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {l[0]:.4f}")
    